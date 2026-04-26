FROM mysql

ENV MYSQL_ROOT_PASSWORD="password"

ADD ic_spider_dbs.sql /docker-entrypoint-initdb.d
