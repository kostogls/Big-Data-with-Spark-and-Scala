import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

import math._
import com.johnsnowlabs.nlp.base.{DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.annotator.{Doc2VecApproach, Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.expressions.Window

object metrics {
  def cosineDiSimilarity(x: Vector, y: Vector): Double = {
    val v1 = x.toArray
    val v2 = y.toArray
    val l1 = scala.math.sqrt(v1.map(x => x * x).sum)
    val l2 = scala.math.sqrt(v2.map(x => x * x).sum)
    val scalar = v1.zip(v2).map(p => p._1 * p._2).sum
    1 - (scalar / (l1 * l2))
  }
}
object Task_4 {

  def main(args: Array[String]) : Unit = {
  /*
  Our goal here is to check if there is a any deviations in parliamentary speeches, in a general level and in 5 major
  parties, level that are products of the economic crisis of 2009. In order to evaluate that hypothesis we use 2 distance/
  similarity metrics. Those metrics are cosine distance between embeddings and euclidean distance. The speeches where
  transformed into embeddings using a Doc2Vec model that we train in the whole corpus (other approaches consists of pre-
  trained embeddings and particularly Bert embeddings. The reason is that they are able to provide a more general
  representation. In this case we avoid them for time and space complexity reasons. Finally, we make the
  assumption that the main political arguments would involve speeches by the leader of each party, due to the fact that
  they are responsible for stating the formal party's opinions.
   */

    val t1 = System.nanoTime
    // Basic Inits
    val ss = SparkSession.builder().master("local[4]").appName("Task_4").getOrCreate()
    import ss.implicits._
    ss.sparkContext.setLogLevel("ERROR")


    //  Reading the dateset and keeping the useful parts
    val inputFile = "C:\\Users\\User\\Downloads\\Greek_Parliament_Proceedings_1989_2020"
    val cols = Seq("member_name","parliamentary_period","parliamentary_session","parliamentary_sitting",
      "government","member_region","member_gender")
    val speechDF =  ss.read.option("header", "true").csv(inputFile)
    val testSpeeces = speechDF.drop(cols:_*).na.drop("any")


    // Keeping only speeches of party leaders as sample of their party policies
    var OnlyPresidents = testSpeeces.filter($"roles".contains("αρχηγος κομματος:")).drop($"roles").persist()
    OnlyPresidents = OnlyPresidents.withColumn("year", year(to_date($"sitting_date","dd/MM/yyyy")))
    OnlyPresidents = OnlyPresidents.withColumn("year",col("year").cast(IntegerType))
      .drop($"sitting_date")
    OnlyPresidents = OnlyPresidents.filter($"year" between(2004,2015))
    //println(OnlyPresidents.count())

    // Create a pipeline for preprocessing and vector embeddings
    // Initialize the pipeline

    val documentAssembler = new DocumentAssembler()
      .setInputCol("speech")
      .setOutputCol("document")
      .setCleanupMode("shrink_full")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

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

    val pipelineModel = pipeline.fit(OnlyPresidents)
    var TransformedOnlyPresidents = pipelineModel.transform(OnlyPresidents)

    val convert2Vector = udf((array : Seq[Float])  => {
      Vectors.dense(array.toArray.map(_.toDouble))
    })

    TransformedOnlyPresidents = TransformedOnlyPresidents.withColumn("str_size",size(flatten($"finished_embeddings")))
      .filter('str_size===100)

    TransformedOnlyPresidents = TransformedOnlyPresidents
      .withColumn("finished_embeddings",convert2Vector(flatten($"finished_embeddings")))


    //TransformedOnlyPresidents.show()
    // Creating a distinct datasets for past and during crisis speeches
    var PreCrisis = TransformedOnlyPresidents.filter($"year" between(2004,2008))
      .drop(Seq("speech","str_size"):_*)
    var DuringCrisis = TransformedOnlyPresidents.filter($"year" between(2009,2015))
      .drop(Seq("speech","str_size"):_*)

    //Aggregating for each party

    PreCrisis = PreCrisis.groupBy($"political_party").agg(Summarizer.max($"finished_embeddings")).persist()
    // Instead of mean also max can be applied

    DuringCrisis = DuringCrisis.groupBy($"political_party").agg(Summarizer.max($"finished_embeddings")).persist()
    // Instead of mean also max can be applied

    // Creating similarity and distance functions

    val cosSimilarity = udf { (x: Vector, y: Vector) =>
      val v1 = x.toArray
      val v2 = y.toArray
      val l1 = scala.math.sqrt(v1.map(x => x*x).sum)
      val l2 = scala.math.sqrt(v2.map(x => x*x).sum)
      val scalar = v1.zip(v2).map(p => p._1*p._2).sum
      scalar/(l1*l2)
    }

    //PreCrisis.show(false)
    //DuringCrisis.show(false)


    val partiesOfInterest = Seq("κομμουνιστικο κομμα ελλαδας","νεα δημοκρατια",
      "συνασπισμος ριζοσπαστικης αριστερας","πανελληνιο σοσιαλιστικο κινημα")
    val preCrisisExample = PreCrisis.filter(col("political_party").isin(partiesOfInterest:_*))
    val duringCrisisExample = DuringCrisis.filter(col("political_party").isin(partiesOfInterest:_*))

    // Similarity between the two time-periods for each party.
    val resultDf = preCrisisExample.join(duringCrisisExample.
      withColumnRenamed("max(finished_embeddings)","preMaxEmb"),Seq("political_party"))
      .withColumn("similarity", cosSimilarity($"max(finished_embeddings)",$"preMaxEmb"))
      .drop(Seq("max(finished_embeddings)","preMaxEmb"):_*)


    resultDf.show(false)

    // Creating a general embedding for the 2 time-periods
    val PreCrisisSumUp = PreCrisis.select(Summarizer.max($"max(finished_embeddings)"))
    val DuringCrisisSumUp = DuringCrisis.select(Summarizer.max($"max(finished_embeddings)"))
    //PreCrisisSumUp.show(false)
    //DuringCrisisSumUp.show(false)

    //Calculating metrics for the two time periods.
    val distance = math.sqrt(Vectors.sqdist(PreCrisisSumUp.first()(0).asInstanceOf[Vector],
     DuringCrisisSumUp.first()(0).asInstanceOf[Vector]))
    println("Euclidean distance between the aggregated vectors of each time window: "+distance)
    println("-----------------------------------------------------------------------")
    val cosine = metrics.cosineDiSimilarity(PreCrisisSumUp.first()(0).asInstanceOf[Vector],
      DuringCrisisSumUp.first()(0).asInstanceOf[Vector])
    println("Cosine distance between the aggregated vectors of each time window: " +  cosine)

    val duration = ((System.nanoTime - t1) / 1e9d)/60
    println(duration)
  }

}