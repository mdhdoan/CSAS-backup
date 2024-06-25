/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// News Articles
//
CREATE CONSTRAINT link_article IF NOT EXISTS FOR (n:Article) REQUIRE n.link IS UNIQUE;
CREATE TEXT INDEX title_article IF NOT EXISTS FOR (n:Article) ON (n.title);
CREATE INDEX pub_date_article IF NOT EXISTS FOR (n:Article) ON (n.pub_date);
CREATE INDEX keywords_article IF NOT EXISTS FOR (n:Article) ON (n.keywords);
//
CREATE TEXT INDEX summary_article IF NOT EXISTS FOR (n:Article) ON (n.summary);
//
//
// Clusters
//
CREATE CONSTRAINT id_cluster IF NOT EXISTS FOR (n:Cluster) REQUIRE n.id IS UNIQUE;
CREATE TEXT INDEX label_cluster IF NOT EXISTS FOR (n:Cluster) ON (n.label);
CREATE TEXT INDEX all_labels_cluster IF NOT EXISTS FOR (n:Cluster) ON (n.all_labels);
CREATE TEXT INDEX summary_cluster IF NOT EXISTS FOR (n:Cluster) ON (n.summary);
CREATE INDEX kbi_keywords_cluster IF NOT EXISTS FOR (n:Cluster) ON (n.kbi_keywords);
CREATE INDEX mmr_keywords_cluster IF NOT EXISTS FOR (n:Cluster) ON (n.mmr_keywords);
CREATE INDEX name_cluster IF NOT EXISTS FOR (n:Cluster) ON (n.name);
//
CREATE CONSTRAINT id_outlier IF NOT EXISTS FOR (n:Outlier) REQUIRE n.id IS UNIQUE;
//
//
// Source
//
CREATE CONSTRAINT id_source IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;
//
//
// Create vector index for Article/Cluster
//
CREATE CONSTRAINT uid_embedding IF NOT EXISTS FOR (n:Embedding) REQUIRE n.uid IS UNIQUE;
//
CREATE VECTOR INDEX `embeddings` IF NOT EXISTS FOR (n:Embedding) ON (n.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 4096,
        `vector.similarity_function`: 'cosine'
    }};
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
