package com.hackthon.household.pushtokafka;

import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.serialization.LongDeserializer;
import org.apache.kafka.common.serialization.StringDeserializer;
import java.util.Collections;
import java.util.Map;
import java.util.Properties;

public class KafkaConsumerFile {

	private final static String TOPIC = "test";
	private final static String BOOTSTRAP_SERVERS = "localhost:9092";
	// private Consumer consumer;

	private static Consumer<Long, String> createConsumer() {
		final Properties props = new Properties();
		props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
		props.put(ConsumerConfig.GROUP_ID_CONFIG, "KafkaExampleConsumer");
		props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, LongDeserializer.class.getName());
		props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
		// Create the consumer using props.
		final Consumer<Long, String> consumer = new KafkaConsumer<>(props);
		// Subscribe to the topic.
		// consumer.subscribe(Collections.singletonList(TOPIC));
		consumer.subscribe(TOPIC);
		return consumer;
	}

	static void runConsumer() throws InterruptedException {
		final Consumer<Long, String> consumer = createConsumer();
		final int giveUp = 100;
		int noRecordsCount = 0;
		while (true) {
			final Map<String, ConsumerRecords<Long, String>> consumerRecords = consumer.poll(1000);
			
			//consumerRecords.get(topic).equals(null);
			if (consumerRecords.count() == 0) {
				noRecordsCount++;
				if (noRecordsCount > giveUp)
					break;
				else
					continue;
			}
			consumerRecords.forEach(record -> {
				System.out.printf("Consumer Record:(%d, %s, %d, %d)\n", record.key(), record.value(),
						record.partition(), record.offset());
			});
			consumer.commitAsync();
			
		}
		consumer.close();
		System.out.println("DONE");
	}

	public static void main(String[] args) throws Exception {
		runConsumer();
		System.out.println("Processing Done !!!");
	}

}