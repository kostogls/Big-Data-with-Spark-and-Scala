import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, IDFModel, Tokenizer}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.sql.functions.{array, asc, col, collect_list, collect_set, concat_ws, count, explode, flatten, lag, last, regexp_replace, size, sort_array, split, to_date, when}
import org.apache.spark.sql.{Column, DataFrame, SparkSession, functions}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.IntegerType
import shapeless.syntax.std.tuple.unitTupleOps

import scala.collection.mutable

object project {

  def get_Names(): List[String] = {
    var num = List("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18")
    return num
  }

  def countNullInCols(columns:Array[String]):Array[Column]={
    columns.map(c=>{
      count(when(col(c).isNull ||
        //col(c)==="" ||
        col(c).contains("NULL") ||
        col(c).contains("null"),c)
      ).alias(c)
    })
  }

  def topTerms(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int, numTerms: Int, termId: Array[String])
  : Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map {
        case (score, id) => (termId(id), score)
      }
    }
    topTerms
  }


  def main(args: Array[String]): Unit = {

    val ss = SparkSession.builder().master("local").appName("App").getOrCreate()

    ss.sparkContext.setLogLevel("ERROR")

/*
    val names = get_Names()
    names.foreach(names => {
      println("Starting file " + names)
      val df = ss.read.parquet("processed.parquet/part-000" + names + "-98b58bde-4095-40d6-abcc-8edc5a0cc0c7-c000.snappy.parquet")
      //val df_ = df.withColumn("clean_period", regexp_replace(col("parliamentary_period"), "period",""))
      //val df__ = df_.withColumn("clean_period",col("clean_period").cast(IntegerType))
      //val sorted = df__.sort(asc("clean_period"))
      // yeah it's sorted now
      df.show()
      val tokens_df = df.filter(size(df("cleaned_speech")) > 0)



      //tokens_df.printSchema()

      //println(tokens_df.count)

      //tokens_df.show()


      val vectorizer = new CountVectorizer()
        .setInputCol("cleaned_speech")
        .setOutputCol("raw")
        .setMaxDF(0.2)
        .setVocabSize(17000)

      val vec = vectorizer.fit(tokens_df)
      val df2 = vec.transform(tokens_df).select("raw")
      df2.show()

      val idf = new IDF().setInputCol("raw").setOutputCol("features")
      val model_idf = idf.fit(df2)
      val df3 = model_idf.transform(df2).select("features")
      df3.show(truncate = false)


      val rdd = df3.select("features").rdd.map {
        row => Vectors.fromML(row.getAs[MLVector]("features"))
      }

      rdd.cache()
      val mat = new RowMatrix(rdd)
      val svd = mat.computeSVD(15, computeU = true)
      print(topTerms(svd, 15, 10, vec.vocabulary).mkString("\n\n"))
    })

*/
    val dataf = ss.read.parquet("processed.parquet")
    val dataf2 = dataf.withColumn("clean_period", regexp_replace(col("parliamentary_period"), "period",""))
    val dataf3 = dataf2.withColumn("clean_period",col("clean_period").cast(IntegerType))
    dataf3.show()
    //dataf.groupBy("member_name".distinct).
    //val d2 = dataf.withColumn("date")
    //d2.show()
    val guessedFraction = 0.2
    val newSample = dataf3.sample(true, guessedFraction).limit(1000)
    val name_speech = newSample.select(col("member_name"), col("cleaned_speech"), col("clean_period"))
    println("namespeech")
    name_speech.show(100)
    val window = Window.partitionBy(col("member_name"), col("clean_period"))

    val d3 = name_speech.withColumn("test", collect_list(("cleaned_speech")).over(window)).groupBy("member_name","clean_period").agg(last("test"))
    d3.printSchema()
    println("d3")
    d3.show(300)
    println(d3.count)


    val d4 = d3.withColumn("flatten", concat_ws(",", flatten(col("last(test)"))))
    println(d4)
    d4.show()
    d4.select(split(col("flatten"),",").as("flatten")).show()

    //val again = d4.select(split(col("flatten"),",").as("flatten"))
    //println("again")
    //again.show()




    //val distinct = dataf.select(dataf("member_name")).distinct()
    //distinct.show()


  }

}