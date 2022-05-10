
// Basic Imports for data handling
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import com.johnsnowlabs.nlp.base.{DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.annotator.{Doc2VecApproach,Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg._

object Task_2 {
  def main(args: Array[String]) : Unit = {
    val t1 = System.nanoTime


    val ss = SparkSession.builder().master("local[*]").appName("Task_2")
      .config("spark.sql.broadcastTimeout", "36000").getOrCreate()
    import ss.implicits._

    ss.sparkContext.setLogLevel("ERROR")
    //Reading files
    val inputFile = "C:\\Users\\User\\Downloads\\Greek_Parliament_Proceedings_1989_2020"

    val cols = Seq("sitting_date","parliamentary_period","parliamentary_session","parliamentary_sitting",
                   "political_party","government","member_region","member_gender")

    val speechDF =  ss.read.option("header", "true").csv(inputFile).sample(0.0001)

    val testSpeeces = speechDF.drop(cols:_*).na.drop("any")

    //var OnlyPresidents = testSpeecesPre.filter($"roles".contains("αρχηγος κομματος:")).drop($"roles")

    //val testSpeeces = OnlyPresidents.limit(100000)//.sample(0.2)
    // Initialize the pipeline

    val documentAssembler = new DocumentAssembler()
      .setInputCol("speech")
      .setOutputCol("document")
      .setCleanupMode("shrink_full")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    /*
    val embeddings = new Doc2VecApproach()
      .setInputCols("token")
      .setOutputCol("embeddings")
    */
    val embeddings = Doc2VecApproach.load("C:\\Users\\User\\Downloads\\parlGRnew.model")
      .setInputCols("token")
      .setOutputCol("embeddings")

    val embeddingsFinisher =  new EmbeddingsFinisher()
      .setInputCols("embeddings")
      .setOutputCols("finished_embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings,
        embeddingsFinisher
      ))

    // Run the pipeline
    val pipelineModel = pipeline.fit(testSpeeces)
    var testResults = pipelineModel.transform(testSpeeces)

    // Cleaning the unnecessary columns

    val new_cols = Seq("document","token", "speech", "embeddings")
    testResults = testResults.drop(new_cols:_*)

    testResults = testResults.withColumn("str_size",size(flatten($"finished_embeddings")))
      .filter('str_size===100)

    testResults = testResults.select($"member_name",flatten($"finished_embeddings"))
    //println(testResults.select($"member_name").distinct().count())

    // Casting for array to vectors in order to aggregate later

    val convert2Vector = udf((array : Seq[Float])  => {
      Vectors.dense(array.toArray.map(_.toDouble))
    })

    testResults = testResults.select("*").withColumn("flatten(finished_embeddings)",
      convert2Vector($"flatten(finished_embeddings)"))

    // Groupby and aggregate by name to give each member a vector

    testResults = testResults.groupBy($"member_name").agg(Summarizer.max($"flatten(finished_embeddings)"))
      .alias("max") // Instead of mean also max can be applied

    testResults = testResults.
      withColumnRenamed("max(flatten(finished_embeddings))","member_vector").persist()
    // Calculate the dot product of the members (cosine similarity)


    val cosSimilarity = udf { (x: Vector, y: Vector) =>
      val v1 = x.toArray
      val v2 = y.toArray
      val l1 = scala.math.sqrt(v1.map(x => x*x).sum)
      val l2 = scala.math.sqrt(v2.map(x => x*x).sum)
      val scalar = v1.zip(v2).map(p => p._1*p._2).sum
      scalar/(l1*l2)
    }

    val temp = testResults.
      withColumnRenamed("member_name","member_name_cross")
      .withColumnRenamed("member_vector","member_vector_cross").persist()

    val similarities = testResults.crossJoin(temp)
      .withColumn("similarity", cosSimilarity($"member_vector",$"member_vector_cross"))
      .drop(Seq("member_vector","member_vector_cross"):_*)

    val k = 10

    val filtered = similarities
      .withColumn("similarity", when(col("similarity").isNaN, 0).otherwise(col("similarity")))
      .withColumn("rank", row_number().over(
        Window.partitionBy(col("member_name")).orderBy(col("similarity").desc)))
      .filter(col("rank")between(2,k+1)).drop($"rank")

    filtered.show(50,false)
    val duration = (System.nanoTime - t1) / 1e9d
    println(duration)
    //similarities.show(10)
  }
}

