#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>

int listenfd, connfd=0;
struct sockaddr_in servaddr,cliaddr;
socklen_t clilen;
struct timeval timeout = {0,100};

void initTCPserver(int port)
{
  listenfd=socket(AF_INET,SOCK_STREAM,0);
  
  bzero(&servaddr,sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr=htonl(INADDR_ANY);
  servaddr.sin_port=htons(32000);
  bind(listenfd,(struct sockaddr *)&servaddr,sizeof(servaddr));
}

int tcpRead(char *data, int datasize)
{
  int n;
  fd_set dataReady;

  if(!connfd) { // there is no client: listen until timeout or accept
    FD_ZERO(&dataReady);
    FD_SET(listenfd,&dataReady);
    if(select(listenfd+1, &dataReady, NULL,NULL, &timeout) == -1) {
      fprintf(stderr,"listen select failed!\n"); exit(1);
    }
    listen(listenfd,1); // listen for one connection at a time
    
    clilen=sizeof(cliaddr);
    if(FD_ISSET(listenfd, &dataReady)) {
      fprintf(stderr,"accepting a client!\n");
      connfd = accept(listenfd,(struct sockaddr *)&cliaddr,&clilen);
    } else {
      //fprintf(stderr,"no client!\n");
      return(0); // no client so no work
    }
  }
  
  if(!connfd) return(0);

  // read the data
  FD_ZERO(&dataReady);
  FD_SET(connfd,&dataReady);
  if(select(connfd+1, &dataReady, NULL,NULL, &timeout) == -1) {
    fprintf(stderr,"data select failed!\n"); exit(1);
  }

  if(FD_ISSET(connfd, &dataReady)) {
    FD_CLR(connfd, &dataReady);
    for(n=0; n < datasize;) {
      int size = ((datasize-n) > SSIZE_MAX)?SSIZE_MAX:(datasize-n);
      int ret = read(connfd, data+n, size);
      if(ret <= 0) break; // error
      n += ret;
    }
    if(n < datasize) {
      fprintf(stderr,"Incomplete read %d bytes %d\n", n, datasize);
      perror("Read failure!");
      close(connfd);
      connfd=0;
      return(0);
    } 
    return(1);
  } else {
    //fprintf(stderr, "read timeout\n");
  }
  return(0);
}


#ifdef USE_MAIN
typedef struct {
   unsigned char r;
   unsigned char g;
   unsigned char b;
} rgb24_t;

int main(int argc, char *argv[])
{
   int width=300, height=240;
   int i,port, datasize;
	
   if(argc < 3) {
      fprintf(stderr,"Use: port widthxheight\n");
      exit(1);
      }

   port=atoi(argv[1]);
   sscanf(argv[2], "%dx%d", &width, &height);
   if(width <= 0 || height <=0 ) {
      fprintf(stderr,"bad width  %d or height %d\n", width, height);
      exit(1);
      }

   initTCPserver(port);

   datasize = width * height * sizeof(rgb24_t);
   rgb24_t *data = (rgb24_t*) malloc( datasize );
   for(;;) {
     if(tcpRead((char*)data, datasize)) { // have data
       /*
	 #pragma omp parallel for
	 for(i=0; i < width*height; i++) {
	 data[i].r = ~data[i].r;
	 data[i].g = ~data[i].g;
	 data[i].b = ~data[i].b;
	 }
       */
       fwrite(data, sizeof(rgb24_t), width*height, stdout);
       fflush(stdout);
     }
   }
}
#endif
