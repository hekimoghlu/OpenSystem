/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/****************************************
 * enable/disable ifdef
*****************************************/
#include "saslauthd-main.h"

#ifdef USE_DOORS_IPC
/****************************************/



/****************************************
 * includes
*****************************************/
#include <door.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stropts.h>

#include "globals.h"
#include "utils.h"
 

/****************************************
 * declarations/protos
 *****************************************/
static void	do_request(void *, char *, size_t, door_desc_t *, uint_t);
static void	send_no(char *);
static void	need_thread(door_info_t*);
static void	*server_thread(void *);

/****************************************
 * module globals
 *****************************************/
static char			*door_file;  /* Path to the door file        */
static int			door_fd;     /* Door file descriptor         */
static pthread_attr_t thread_attr;	     /* Thread attributes            */
static int			num_thr;     /* Number of threads            */
static pthread_mutex_t		num_lock;    /* Lock for update              */

/****************************************
 * flags       	global from saslauthd-main.c
 * run_path    	global from saslauthd-main.c
 * num_procs   	global from saslauthd-main.c
 * detach_tty()	function from saslauthd-main.c
 * logger()		function from utils.c
 *****************************************/

/*************************************************************
 * IPC init. Initialize the environment specific to the 
 * Sun doors IPC method.
 *
 * __Required Function__
 **************************************************************/
void ipc_init() {
	int	rc;
	size_t  door_file_len;

	/**************************************************************
         * Doors detach immediately, otherwise the process gets confused.
         * (they don't follow fork() properly)
	 **************************************************************/
	detach_tty();

	/**************************************************************
	 * Setup the door file and the door.
	 **************************************************************/
	door_file_len = strlen(run_path) + sizeof(DOOR_FILE) + 1;
	if (!(door_file = malloc(door_file_len))) {
		logger(L_ERR, L_FUNC, "could not allocate memory");
		exit(1);
	}

	strlcpy(door_file, run_path, door_file_len);
	strlcat(door_file, DOOR_FILE, door_file_len);
	unlink(door_file);

	if ((door_fd = open(door_file, O_CREAT|O_RDWR|O_TRUNC, 0666)) == -1) {
		rc = errno;
		logger(L_ERR, L_FUNC, "could not open door file: %s",
		       door_file);
		logger(L_ERR, L_FUNC, "open: %s", strerror(rc));
		exit(1);
	}

	close(door_fd);

	if ((door_fd = door_create(&do_request, NULL, 0)) < 0) {
		logger(L_ERR, L_FUNC, "failed to create door");
		exit(1);
	}

	door_server_create(&need_thread);

	if (fattach(door_fd, door_file) < 0) {
		logger(L_ERR, L_FUNC, "failed to attach door to file: %s",
		       door_file);
		exit(1);
	}

	if (chmod(door_file, 0644) < 0) {
		rc = errno;
		logger(L_ERR, L_FUNC, "failed to chmod door file: %s",
		       door_file);
		logger(L_ERR, L_FUNC, "chmod: %s", strerror(rc));
		exit(1);
	}

	logger(L_INFO, L_FUNC, "door on: %s", door_file);

	/**************************************************************
	 * The doors api will handle threads for us, clear the process 
	 * model global flag.
	 **************************************************************/
	flags &= ~USE_PROCESS_MODEL;

 	/* Initialize mutex */
	pthread_mutex_init(&num_lock, NULL);

	/* Initialize thread attributes */
	pthread_attr_init(&thread_attr);
	pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_DETACHED);

	return;
}


/*************************************************************
 * Main IPC loop. Sit idle waiting for a door request. All
 * request get routed to do_request() via the doors api.
 *
 * __Required Function__
 **************************************************************/
void ipc_loop() {
	while(1) {
		pause();
	}

	return;
}


/*************************************************************
 * General cleanup. Unlink our files.
 *
 * __Required Function__
 **************************************************************/
void ipc_cleanup() {
	unlink(door_file);

	if (flags & VERBOSE)
		logger(L_DEBUG, L_FUNC, "door file removed: %s", door_file);
}


/*************************************************************
 * Handle the door data, pass the request off to
 * do_auth() back in saslauthd-main.c, then send the 
 * result back through the door.
 **************************************************************/
void do_request(void *cookie, char *data, size_t datasize, door_desc_t *dp, size_t ndesc) {
	unsigned short		count = 0;                 /* input/output data byte count           */
	char			*response = NULL;          /* response to send to the client         */
	char			response_buff[1024];       /* temporary response buffer              */
	char			*dataend;                  /* EOD marker for the door data           */
	char			login[MAX_REQ_LEN + 1];    /* account name to authenticate           */
	char			password[MAX_REQ_LEN + 1]; /* password for authentication            */
	char			service[MAX_REQ_LEN + 1];  /* service name for authentication        */
	char			realm[MAX_REQ_LEN + 1];    /* user realm for authentication          */


	/**************************************************************
	 * The input data string consists of the login id, password,
	 * service name and user realm. We'll break them up and then
	 * authenticate them.
	 **************************************************************/
	dataend = data + datasize;

	/* login id */
	memcpy(&count, data, sizeof(unsigned short));

	count = ntohs(count);
	data += sizeof(unsigned short);

	if (count > MAX_REQ_LEN || data + count > dataend) {
		logger(L_ERR, L_FUNC, "login exceeds MAX_REQ_LEN: %d",
		       MAX_REQ_LEN);
		send_no("");
		return;
	}	

	memcpy(login, data, count);
	login[count] = '\0';
	data += count;

	/* password */
	memcpy(&count, data, sizeof(unsigned short));

	count = ntohs(count);
	data += sizeof(unsigned short);

	if (count > MAX_REQ_LEN || data + count > dataend) {
		logger(L_ERR, L_FUNC, "password exceeds MAX_REQ_LEN: %d",
		       MAX_REQ_LEN);
		send_no("");
		return;
	}	

	memcpy(password, data, count);
	password[count] = '\0';
	data += count;

	/* service */
	memcpy(&count, data, sizeof(unsigned short));

	count = ntohs(count);
	data += sizeof(unsigned short);

	if (count > MAX_REQ_LEN || data + count > dataend) {
		logger(L_ERR, L_FUNC, "service exceeds MAX_REQ_LEN: %d",
		       MAX_REQ_LEN);
		send_no("");
		return;
	}	

	memcpy(service, data, count);
	service[count] = '\0';
	data += count;

	/* realm */
	memcpy(&count, data, sizeof(unsigned short));

	count = ntohs(count);
	data += sizeof(unsigned short);

	if (count > MAX_REQ_LEN || data + count > dataend) {
		logger(L_ERR, L_FUNC, "realm exceeds MAX_REQ_LEN: %d",
		       MAX_REQ_LEN);
		send_no("");
		return;
	}	

	memcpy(realm, data, count);
	realm[count] = '\0';

	/**************************************************************
 	 * We don't allow NULL passwords or login names
	 **************************************************************/
	if (*login == '\0') {
		logger(L_ERR, L_FUNC, "NULL login received");
		send_no("NULL login received");
		return;
	}	
	
	if (*password == '\0') {
		logger(L_ERR, L_FUNC, "NULL password received");
		send_no("NULL password received");
		return;
	}	

	/**************************************************************
	 * Get the mechanism response from do_auth() and send it back.
	 **************************************************************/
	response = do_auth(login, password, service, realm);

	memset(password, 0, strlen(password));

	if (response == NULL) {
	    send_no("NULL response from mechanism");
	    return;
	}	

	strncpy(response_buff, response, 1023);
	response_buff[1023] = '\0';
	free(response);

	if (flags & VERBOSE)
	    logger(L_DEBUG, L_FUNC, "response: %s", response_buff);

	if(door_return(response_buff, strlen(response_buff), NULL, 0) < 0)
	    logger(L_ERR, L_FUNC, "door_return: %s", strerror(errno));

	return;
}

/*************************************************************
 * The available server  thread  pool  is  depleted.
 * Create a new thread with suitable attributes.
 * Client door_call() will block until server thread is available.
 **************************************************************/
void need_thread(door_info_t *di) {
    pthread_t newt;
    int more;
    
    if (num_procs > 0) {
	pthread_mutex_lock(&num_lock);
	more = (num_thr < num_procs);
	if (more) num_thr++;
	pthread_mutex_unlock(&num_lock);
	if (!more) return;
    }

    pthread_create(&newt, &thread_attr, &server_thread, NULL);
}
 
/*************************************************************
 * Start a new server thread.
 * Make it available for door invocations.
 **************************************************************/
void *server_thread(void *arg) {
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
    door_return(NULL, 0, NULL, 0);
}

/*************************************************************
 * In case something went out to lunch while parsing the
 * request data, we may want to attempt to send back a
 * "NO" response through the door. The mesg is optional.
 **************************************************************/
void send_no(char *mesg) {
	char		buff[1024];

	buff[0] = 'N';
	buff[1] = 'O';
	buff[2] = ' ';

	/* buff, except for the trailing NUL and 'NO ' */
	strncpy(buff + 3, mesg, sizeof(buff) - 1 - 3);
	buff[1023] = '\0';

	if (flags & VERBOSE)
	    logger(L_DEBUG, L_FUNC, "response: %s", buff);

	if(door_return(buff, strlen(buff), NULL, 0) < 0)
	    logger(L_ERR, L_FUNC, "door_return: %s", strerror(errno));

	return;	
}

#endif /* USE_DOORS_IPC */
