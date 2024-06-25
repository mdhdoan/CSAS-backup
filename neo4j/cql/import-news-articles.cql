/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Import News Articles
//
WITH ['cpaws.org.jsonl', 'hakaimagazine.com.jsonl', 'ocean.org.jsonl', 'wwf.ca.jsonl'] AS source_files
    UNWIND source_files AS source_file
WITH source_file
    CALL apoc.periodic.iterate('
	    CALL apoc.load.json("file:///" + $source_file) YIELD value RETURN value AS doc
    ','
        WITH doc
            MERGE (n:Article {link: doc.metadata.link})
                SET 
                    n.pub_name = REPLACE($source_file, ".jsonl", ""),
                    n.keywords = doc.metadata.keywords,
                    n.title = doc.metadata.title,
                    n.summary = doc.metadata.summary,
                    n.content = doc.page_content
        RETURN doc.metadata.link AS url
    ',
    {params: {source_file: source_file}, batchSize:10, parallel:true})
    YIELD total RETURN total;
//
// WITH ['SFR-Embedding-2_R-emb.jsonl'] AS source_files
WITH ['SFR-Embedding-Mistral-emb.jsonl'] AS source_files
// WITH ['all-MiniLM-L6-v2-emb.jsonl'] AS source_files
    UNWIND source_files AS source_file
WITH source_file
    CALL apoc.periodic.iterate('
	    CALL apoc.load.json("file:///" + $source_file) YIELD value RETURN value AS doc
    ','
        WITH doc
            MERGE (a:Article {link: doc.link})
            MERGE (e:Embedding {uid: doc.link})
            MERGE (a)<-[:EMBBEDINGS_OF]-(e)
        WITH doc, e
            CALL db.create.setNodeVectorProperty(e, "embedding", doc.embedding)
        RETURN e.uid AS url
    ',
    {params: {source_file: source_file}, batchSize:10, parallel:true})
    YIELD total RETURN total;
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Import Article Clusters
//
// WITH ['SFR-Embedding-2_R-cls.jsonl'] AS source_files
WITH ['SFR-Embedding-Mistral-cls.jsonl'] AS source_files
// WITH ['all-MiniLM-L6-v2-cls.jsonl'] AS source_files
    UNWIND source_files AS source_file
WITH source_file
    CALL apoc.periodic.iterate('
        CALL apoc.load.json("file:///" + $source_file) YIELD value AS topic
        RETURN topic
    ','
        WITH topic
            MERGE (c:Cluster {id: topic.id})
                SET c.name = topic.name,
                    c.label = topic.label,
                    c.all_label = topic.all_labels,
                    c.summary = topic.summary,
                    c.keywords = apoc.coll.sort(apoc.coll.toSet(topic.keywords + topic.KBI + topic.MMR)),
                    c.rep_keywords = apoc.coll.sort(topic.keywords),
                    c.kbi_keywords = apoc.coll.sort(topic.KBI),
                    c.mmr_keywords = apoc.coll.sort(topic.MMR)
        WITH topic, c
            FOREACH (rep IN topic.representative_docs |
                MERGE (a:Article {link: rep[1]})
                MERGE (c)<-[:REPRESENTS {prob: rep[2]}]-(a)
            )
            FOREACH (sub_topic_id IN topic.sub_topics |
                MERGE (s:Cluster {id: sub_topic_id})
                MERGE (c)<-[r:PART_OF]-(s)
            )
            FOREACH (dummy IN CASE WHEN topic.sub_topics IS NULL OR SIZE(topic.sub_topics) = 0 THEN [1] ELSE [] END |
                FOREACH (article IN topic.articles |
                    MERGE (a:Article {link: article[2]})
                    MERGE (c)<-[:MEMBER_OF {prob: article[3]}]-(a)
                )
            )
        RETURN c.id
    ',
    {params: {source_file: source_file}, batchSize:100, parallel:false})
    YIELD total RETURN total;
//
// WITH ['SFR-Embedding-2_R-cls.jsonl'] AS source_files
WITH ['SFR-Embedding-Mistral-cls.jsonl'] AS source_files
// WITH ['all-MiniLM-L6-v2-cls.jsonl'] AS source_files
    UNWIND source_files AS source_file
WITH source_file
    CALL apoc.periodic.iterate('
        CALL apoc.load.json("file:///" + $source_file) YIELD value AS topic
        RETURN topic
    ','
        WITH topic
            WHERE topic.embedding IS NOT NULL
            MERGE (c:Cluster {id: topic.id})
            MERGE (e:Embedding {uid: topic.id})
            MERGE (c)<-[:EMBBEDINGS_OF]-(e)
        WITH topic, e
            CALL db.create.setNodeVectorProperty(e, "embedding", topic.embedding)
        RETURN e.uid
    ',
    {params: {source_file: source_file}, batchSize:100, parallel:false})
    YIELD total RETURN total;
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Import Article Clusters
//
WITH ['cpaws.org', 'hakaimagazine.com', 'ocean.org', 'wwf.ca'] AS source_ids
    UNWIND source_ids AS source_id
WITH source_id
    MERGE (s:Source {id: source_id})
RETURN s.id;
//
MATCH (n:Article)
WITH n
    MATCH (s:Source {id: n.pub_name})
    MERGE (n)<-[:FROM]-(s)
RETURN COUNT(n);
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Import Article Clusters
MATCH (c:Cluster) WHERE c.label IS NULL AND c.summary = 'Outliers' SET c.label = 'Outliers', c.all_labels = ['Outliers'];
//
MATCH (a:Article) SET a.content = REPLACE(a.content, "Article body copy", "")
//
MATCH (a:Article) SET a.content = REPLACE(a.content, "Article body copy", "")
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
