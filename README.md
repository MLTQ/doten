# doten
Better Dota Analytics


How to get replay CSVs:
1. get java & maven, set JAVA_HOME and MAVEN_HOME paths
2. get https://github.com/odota/parser and build with mvn /path/to/parser/jar (name) package
3. run the odota server
4. curl localhost:5600 --data-binary "@/mnt/c/Users/aweso/Documents/Code/doten/replays/6464136573.dem" -o "6464136573.json"
5. run the python parser