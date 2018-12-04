wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute01.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute02.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute03.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute04.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute05.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute06.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute07.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute08.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute09.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute10.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute11.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute12.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute13.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute14.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute15.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute16.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute17.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute18.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute19.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute20.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute21.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute22.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute23.zip && \
wget http://www.iro.umontreal.ca/~labimage/Dataset/chute-zip/chute24.zip && \
wget -d --user-agent="Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0" http://le2i.cnrs.fr/spip.php\?action\=acceder_document\&arg\=498\&cle\=b277f8e6bf6e1aa6a726e8692942a1d6ef05d375\&file\=zip%2FOffice.zip &&\
wget -d --user-agent="Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0" "http://le2i.cnrs.fr/spip.php?action=acceder_document&arg=498&cle=b277f8e6bf6e1aa6a726e8692942a1d6ef05d375&file=zip%2FOffice.zip" &&\
wget -d --user-agent="Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0" "http://le2i.cnrs.fr/spip.php?action=acceder_document&arg=500&cle=bd4a06f0a4dc56cee7e2534261b832ac8850e557&file=zip%2FLecture_room.zip" &&\
wget -d --user-agent="Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0" "http://le2i.cnrs.fr/spip.php?action=acceder_document&arg=511&cle=c9562e2d3ded9cf5f1acc8cf702b5d919a7d03d0&file=zip%2FHome_01.zip" &&\
wget -d --user-agent="Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0" "http://le2i.cnrs.fr/spip.php?action=acceder_document&arg=512&cle=d1f28718ccfba1c59aaced7db69897163cb48d5d&file=zip%2FCoffee_room_01.zip" &&\
wget -d --user-agent="Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0" "http://le2i.cnrs.fr/spip.php?action=acceder_document&arg=513&cle=d2f08c8cb5aae493f41e9f05c027a5ff7e5479f6&file=zip%2FCoffee_room_02.zip" &&\
sudo apt-get install unp && \
unp *.zip &&\
rm -rf *.zip
