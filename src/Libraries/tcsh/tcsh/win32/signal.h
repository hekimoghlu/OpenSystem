/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
// signal.h: signal emulation things
// -amol
//
#ifndef SIGNAL_H
#define SIGNAL_H


#define NSIG 23     

// These must be CTRL_xxx_EVENT+1 (in wincon.h)
//
#define SIGINT		1 
#define SIGBREAK 	2
#define SIGHUP		3 //CTRL_CLOSE_EVENT
// 3 and 4 are reserved. hence we can't use 4 and 5
#define	SIGTERM		6 // ctrl_logoff
#define SIGKILL		7 // ctrl_shutdown

#define SIGILL		8 
#define SIGFPE		9	
#define SIGALRM		10
//#define SIGWINCH	11
#define SIGSEGV 	12	
#define SIGSTOP 	13
#define SIGPIPE 	14
#define SIGCHLD 	15
#define SIGCONT		16 
#define SIGTSTP 	18
#define SIGTTOU 	19
#define SIGTTIN 	20
#define SIGABRT 	22	

#define SIGQUIT SIGBREAK

/* signal action codes */

#define SIG_DFL (void (*)(int))IntToPtr(0)   /* default signal action */
#define SIG_IGN (void (*)(int))IntToPtr(1)   /* ignore signal */
#define SIG_SGE (void (*)(int))IntToPtr(3)   /* signal gets error */
#define SIG_ACK (void (*)(int))IntToPtr(4)   /* acknowledge */


/* signal error value (returned by signal call on error) */

#define SIG_ERR (void (*)(int))IntToPtr(-1)   /* signal error value */


#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2

#undef signal
#define signal _nt_signal

typedef unsigned long sigset_t;
typedef void Sigfunc (int);

struct sigaction {
	Sigfunc *sa_handler;
	sigset_t sa_mask;
	int sa_flags;
};


#define sigemptyset(ptr) (*(ptr) = 0)
#define sigfillset(ptr)  ( *(ptr) = ~(sigset_t)0,0)


/* Function prototypes */

void (* _nt_signal(int, void (*)(int)))(int);

int sigaddset(sigset_t*, int);
int sigdelset(sigset_t*,int);
unsigned int alarm(unsigned int);

int sigismember(const sigset_t *set, int);
int sigprocmask(int ,const sigset_t*,sigset_t*);
int sigaction(int, const struct sigaction *, struct sigaction*);
int sigsuspend(const sigset_t *sigmask);

#define WNOHANG 0
#define WUNTRACED 1

#define WIFEXITED(a) 1
#define WEXITSTATUS(a) (a)
//#define WIFSIGNALED(a) ((a!= -1)&&(((((unsigned long)(a)) >>24) & 0xC0)!=0))
#define WIFSIGNALED(a) ((a !=-1)&&((((unsigned long)(a)) & 0xC0000000 ) != 0))
#define WTERMSIG(a) (((unsigned long)(a))==0xC000013AL?SIGINT:SIGSEGV)
#define WCOREDUMP(a) 0
#define WIFSTOPPED(a) 0
#define WSTOPSIG(a) 0

int waitpid(pid_t, int*,int);
int times(struct tms*);
  
#endif SIGNAL_H
