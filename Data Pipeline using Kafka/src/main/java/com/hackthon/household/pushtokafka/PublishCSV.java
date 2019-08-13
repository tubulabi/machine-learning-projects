package com.hackthon.household.pushtokafka;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Properties;

import kafka.javaapi.producer.Producer;
import kafka.producer.KeyedMessage;
import kafka.producer.ProducerConfig;

public class PublishCSV {

	private static ProducerConfig producerConfig;
	private Producer<Integer, String> producer;
	private static final String topic ="test";
	

	public void initialize(){
		
		 Properties props = new Properties();
		 props.put("bootstrap.servers", "localhost:9092");
		 props.put("metadata.broker.list", "localhost:9092");
		// props.put("serializer.class", "kafka.serializer.DefaultEncoder");
		 props.put("request.required.acks", "1");
		
		props.put("retries", 0);
		props.put("batch.size", 16384);
		props.put("linger.ms", 1);
		props.put("buffer.memory", 33554432);
		props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

		 //Map<String, Object> producerConfig = (Map<String, Object>) new ProducerConfig(props);
		 producerConfig = new ProducerConfig(props);
		 producer = new Producer<Integer, String>(producerConfig);
		
		 
	}
	
	//@SuppressWarnings("unchecked")
	public void publishMessage() throws Exception {
		
		String fl ="resource/household.csv";
		File file = new File(fl);
		FileInputStream fstream = new FileInputStream(file);
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
		
		String msg = null;
		
		while((msg = br.readLine()) != null)
		{
			System.out.println(msg);
			KeyedMessage<Integer, String> keyedMessage = new KeyedMessage<Integer, String>(topic, msg);
			//producer.send(keyedMessage);
			producer.send(keyedMessage);
			
		}	
			br.close();
			//producer.close();
		
	}
	
	public static void main(String[] args) throws Exception{
		
         PublishCSV kafkaProducer = new PublishCSV();
         kafkaProducer.initialize();
         kafkaProducer.publishMessage();
	}
}
