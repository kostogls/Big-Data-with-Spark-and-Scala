name := "spark2demo"

version := "0.1"

scalaVersion := "2.12.8"

val sparkVersion = "3.0.1"

resolvers += "SparkPackages" at "https://dl.bintray.com/spark-packages/maven"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkVersion

)