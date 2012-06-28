/*
Copyright (c) 2011 Yahoo! Inc.  All rights reserved.  The copyrights
embodied in the content of this file are licensed under the BSD
(revised) open source license

This creates a binary tree topology over a set of n nodes that connect.

 */
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <netdb.h>
#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <map>

using namespace std;

struct client {
  uint32_t client_ip;
  int socket;
};

struct partial {
  client* nodes;
  size_t filled;
};

static int socket_sort(const void* s1, const void* s2) {
  
  client* socket1 = (client*)s1;
  client* socket2 = (client*)s2;
  return socket1->client_ip - socket2->client_ip;
}

int build_tree(int*  parent, uint16_t* kid_count, int source_count, int offset) {

  if(source_count == 1) {
    kid_count[offset] = 0;
    return offset;
  }
    
  int height = (int)floor(log((double)source_count)/log(2.0));
  int root = (1 << height) - 1;
  int left_count = root;
  int left_offset = offset;
  int left_child = build_tree(parent, kid_count, left_count, left_offset);
  int oroot = root+offset;
  parent[left_child] = oroot;
  
  int right_count = source_count - left_count - 1;
  if (right_count > 0)
    {
      int right_offset = oroot+1;
      
      int right_child = build_tree(parent, kid_count, right_count, right_offset);
      parent[right_child] = oroot;
      kid_count[oroot] = 2;
    }
  else
      kid_count[oroot] = 1;

  return oroot;
}

void fail_write(int fd, const void* buf, size_t count)
{
  if (write(fd,buf,count)==-1)
    {
      cerr << "write failed!" << endl;
      exit(1);
    }
}

int main(int argc, char* argv[]) {
  if (argc > 2)
    {
      cout << "usage: spanning_tree [pid_file]" << endl;
      exit(0);
    }

  int sock = socket(PF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    cerr << "can't open socket!" << endl;
    exit(1);
  }

  int on = 1;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&on, sizeof(on)) < 0) 
    perror("setsockopt SO_REUSEADDR");

  sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  short unsigned int port = 26543;
      
  address.sin_port = htons(port);
  if (bind(sock,(sockaddr*)&address, sizeof(address)) < 0)
    {
      cerr << "failure to bind!" << endl;
      exit(1);
    }

  if (daemon(1,1))
    {
      cerr << "failure to background!" << endl;
      exit(1);
    }

  if (argc == 2)
    {	  
      ofstream pid_file;
      pid_file.open(argv[1]);
      if (!pid_file.is_open())
	{
	  cerr << "error writing pid file" << endl;
	  exit(1);
	}
      pid_file << getpid() << endl;
      pid_file.close();
    }
  
  map<int, partial> partial_nodesets;
  
  while(true) {
    listen(sock, 1024);
    
    sockaddr_in client_address;
    socklen_t size = sizeof(client_address);
    int f = accept(sock,(sockaddr*)&client_address,&size);
    if (f < 0)
      {
	cerr << "bad client socket!" << endl;
	exit (1);
      }

    size_t nonce = 0;
    if (read(f, &nonce, sizeof(nonce)) != sizeof(nonce))
      {
	cerr << "nonce read failed, exiting" << endl;
	exit(1);
      }
    size_t total = 0;
    if (read(f, &total, sizeof(total)) != sizeof(total))
      {
	cerr << "total node count read failed, exiting" << endl;
	exit(1);
      }
    size_t id = 0;
    if (read(f, &id, sizeof(id)) != sizeof(id))
      {
	cerr << "node id read failed, exiting" << endl;
	exit(1);
      }

    int ok = true;
    if ( id >= total ) 
      {
	cout << "invalid id! " << endl;
	ok = false;
      }
    partial partial_nodeset;
    
    if (partial_nodesets.find(nonce) == partial_nodesets.end() )
      {
	partial_nodeset.nodes = (client*) calloc(total, sizeof(client));
	for (size_t i = 0; i < total; i++)
	  partial_nodeset.nodes[i].client_ip = (uint32_t)-1;
	partial_nodeset.filled = 0;
      }    
    else {
      partial_nodeset = partial_nodesets[nonce];
      partial_nodesets.erase(nonce);
    }

    if (ok && partial_nodeset.nodes[id].client_ip != (uint32_t)-1)
      ok = false;
    fail_write(f,&ok, sizeof(ok));

    if (ok)
      {
	partial_nodeset.nodes[id].client_ip = client_address.sin_addr.s_addr;
	partial_nodeset.nodes[id].socket = f;
	partial_nodeset.filled++;
      }
    if (partial_nodeset.filled != total) //Need to wait for more connections
      {
	partial_nodesets[nonce] = partial_nodeset;
      }
    else
      {//Time to make the spanning tree
	qsort(partial_nodeset.nodes, total, sizeof(client), socket_sort);
	
	int* parent = (int*)calloc(total,sizeof(int));	
	uint16_t* kid_count = (uint16_t*)calloc(total,sizeof(uint16_t));
	
	int root = build_tree(parent, kid_count, total, 0);
	parent[root] = -1;
	
	for (size_t i = 0; i < total; i++)
	  {
	    fail_write(partial_nodeset.nodes[i].socket, &kid_count[i], sizeof(kid_count[i]));
	  }	

	uint16_t* client_ports=(uint16_t*)calloc(total,sizeof(uint16_t));

	for(size_t i = 0;i < total;i++) {
	  int done = 0;
	  if(read(partial_nodeset.nodes[i].socket, &(client_ports[i]), sizeof(client_ports[i])) < (int) sizeof(client_ports[i])) 
	    cerr<<" Port read failed for node "<<i<<" read "<<done<<endl;
	}// all clients have bound to their ports.
	
	for (size_t i = 0; i < total; i++)
	  {
	    if (parent[i] >= 0)
	      {
		fail_write(partial_nodeset.nodes[i].socket, &partial_nodeset.nodes[parent[i]].client_ip, sizeof(partial_nodeset.nodes[parent[i]].client_ip));
		fail_write(partial_nodeset.nodes[i].socket, &client_ports[parent[i]], sizeof(client_ports[parent[i]]));
	      }
	    else
	      {
		int bogus = -1;
		uint32_t bogus2 = -1;
		fail_write(partial_nodeset.nodes[i].socket, &bogus2, sizeof(bogus2));
		fail_write(partial_nodeset.nodes[i].socket, &bogus, sizeof(bogus));
	      }
	    close(partial_nodeset.nodes[i].socket);
	  }
	free (partial_nodeset.nodes);
      }
  }
}
