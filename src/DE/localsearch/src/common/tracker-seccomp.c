/*
 * Copyright (C) 2016, Red Hat Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include "config-miners.h"

#include "tracker-seccomp.h"

#ifdef HAVE_LIBSECCOMP

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <fcntl.h>

#include <linux/netlink.h>

#ifdef HAVE_BTRFS_IOCTL
#include <linux/btrfs.h>
#endif

#include <seccomp.h>

#include "valgrind.h"

#ifndef SYS_SECCOMP
#define SYS_SECCOMP 1
#endif

#define ALLOW_RULE(call) G_STMT_START { \
	int allow_rule_syscall_number = seccomp_syscall_resolve_name (#call); \
	current_syscall = #call; \
	if (allow_rule_syscall_number == __NR_SCMP_ERROR || \
	    seccomp_rule_add (ctx, SCMP_ACT_ALLOW, allow_rule_syscall_number, 0) < 0) \
		goto out; \
} G_STMT_END

#define ERROR_RULE(call, error) G_STMT_START { \
	int error_rule_syscall_number = seccomp_syscall_resolve_name (#call); \
	current_syscall = #call; \
	if (error_rule_syscall_number == __NR_SCMP_ERROR || \
	    seccomp_rule_add (ctx, SCMP_ACT_ERRNO (error), error_rule_syscall_number, 0) < 0) \
		goto out; \
} G_STMT_END

#define CUSTOM_RULE(call, action, arg1) G_STMT_START { \
	int custom_rule_syscall_number = seccomp_syscall_resolve_name (#call); \
	current_syscall = #call; \
	if (custom_rule_syscall_number == __NR_SCMP_ERROR || \
	    seccomp_rule_add (ctx, action, custom_rule_syscall_number, 1, arg1) < 0) \
		goto out; \
} G_STMT_END

#define CUSTOM_RULE_2ARG(call, action, arg1, arg2) G_STMT_START {	  \
	int custom_rule_syscall_number = seccomp_syscall_resolve_name (#call); \
	current_syscall = #call; \
	if (custom_rule_syscall_number == __NR_SCMP_ERROR || \
	    seccomp_rule_add (ctx, action, custom_rule_syscall_number, 2, arg1, arg2) < 0) \
		goto out; \
} G_STMT_END

static void
sigsys_handler (gint       signal,
                siginfo_t *info,
                gpointer   context)
{
	if (info->si_signo == SIGSYS &&
	    info->si_code == SYS_SECCOMP) {
		char *syscall_name;

		syscall_name = seccomp_syscall_resolve_num_arch (SCMP_ARCH_NATIVE,
		                                                 info->si_syscall);
		g_printerr ("Disallowed syscall \"%s\" caught in sandbox\n", syscall_name);
		free (syscall_name);

		/* Ensure to propagate SIGSYS to generate a core file.
		 * Use syscall instead of tgkill because not all libc's
		 * implement the wrappers
		 */
		syscall (SYS_tgkill, gettid(), getpid(), SIGSYS);
	}
}

static gboolean
initialize_sigsys_handler (void)
{
	struct sigaction act = { 0 };

	sigemptyset (&act.sa_mask);
	sigaddset (&act.sa_mask, SIGSYS);
	act.sa_flags = SA_SIGINFO | SA_RESETHAND;
	act.sa_sigaction = sigsys_handler;

	if (sigaction (SIGSYS, &act, NULL) < 0)
		return FALSE;

	return TRUE;
}

gboolean
tracker_seccomp_init (void)
{
	scmp_filter_ctx ctx;
	const gchar *current_syscall = NULL;

	if (RUNNING_ON_VALGRIND) {
		g_message ("Running under Valgrind, Seccomp was disabled");
		return TRUE;
	}

	if (!initialize_sigsys_handler ())
		return FALSE;

	ctx = seccomp_init (SCMP_ACT_TRAP);
	if (ctx == NULL)
		return FALSE;

	/* Memory management */
	ALLOW_RULE (brk);
	ALLOW_RULE (get_mempolicy);
	ALLOW_RULE (set_mempolicy);
	ALLOW_RULE (mmap);
	ALLOW_RULE (mmap2);
	ALLOW_RULE (munmap);
	ALLOW_RULE (mremap);
	ALLOW_RULE (mprotect);
	ALLOW_RULE (madvise);
	ALLOW_RULE (mbind);
	ALLOW_RULE (membarrier);
	ERROR_RULE (mlock, EPERM);
	ERROR_RULE (mlock2, EPERM);
	ERROR_RULE (munlock, EPERM);
	ERROR_RULE (mlockall, EPERM);
	ERROR_RULE (munlockall, EPERM);
	/* Process management */
	ALLOW_RULE (exit_group);
	ALLOW_RULE (getuid);
	ALLOW_RULE (getuid32);
	ALLOW_RULE (getgid);
	ALLOW_RULE (getgid32);
	ALLOW_RULE (getegid);
	ALLOW_RULE (getegid32);
	ALLOW_RULE (geteuid);
	ALLOW_RULE (geteuid32);
	ALLOW_RULE (getppid);
	ALLOW_RULE (gettid);
	ALLOW_RULE (getpid);
	ALLOW_RULE (exit);
	ALLOW_RULE (getrusage);
	ALLOW_RULE (getrlimit);
	ERROR_RULE (sched_getattr, EPERM);
	/* Basic filesystem access */
	ALLOW_RULE (fstat);
	ALLOW_RULE (fstat64);
	ALLOW_RULE (fstatat64);
	ALLOW_RULE (newfstatat);
	ALLOW_RULE (stat);
	ALLOW_RULE (stat64);
	ALLOW_RULE (statfs);
	ALLOW_RULE (statfs64);
	ALLOW_RULE (lstat);
	ALLOW_RULE (lstat64);
	ALLOW_RULE (statx);
	ALLOW_RULE (fstatfs);
	ALLOW_RULE (fstatfs64);
	ALLOW_RULE (access);
	ALLOW_RULE (faccessat);
	ALLOW_RULE (faccessat2);
	ALLOW_RULE (getdents);
	ALLOW_RULE (getdents64);
	ALLOW_RULE (getcwd);
	ALLOW_RULE (readlink);
	ALLOW_RULE (readlinkat);
	ALLOW_RULE (utime);
	ALLOW_RULE (time);
	ALLOW_RULE (fsync);
	ALLOW_RULE (umask);
	ALLOW_RULE (chdir);
	ERROR_RULE (fchown, EPERM);
	ERROR_RULE (fchmod, EPERM);
	ERROR_RULE (chmod, EPERM);
	ERROR_RULE (mkdir, EPERM);
	ERROR_RULE (mkdirat, EPERM);
	ERROR_RULE (rename, EPERM);
	ERROR_RULE (unlink, EPERM);
	/* Processes and threads */
	ALLOW_RULE (clone);
	ALLOW_RULE (clone3);
	ALLOW_RULE (futex);
	ALLOW_RULE (futex_time64);
	ALLOW_RULE (set_robust_list);
	ALLOW_RULE (rseq);
	ALLOW_RULE (rt_sigaction);
	ALLOW_RULE (rt_sigprocmask);
	ALLOW_RULE (rt_sigreturn);
	ALLOW_RULE (sched_yield);
	ALLOW_RULE (sched_getaffinity);
	ALLOW_RULE (sched_get_priority_max);
	ALLOW_RULE (sched_get_priority_min);
	ALLOW_RULE (sched_setattr);
	ALLOW_RULE (nanosleep);
	ALLOW_RULE (clock_nanosleep);
	ALLOW_RULE (clock_nanosleep_time64);
	ALLOW_RULE (waitid);
	ALLOW_RULE (waitpid);
	ALLOW_RULE (wait4);
	ALLOW_RULE (restart_syscall);
	/* Main loops */
	ALLOW_RULE (poll);
	ALLOW_RULE (ppoll);
	ALLOW_RULE (ppoll_time64);
	ALLOW_RULE (fcntl);
	ALLOW_RULE (fcntl64);
	ALLOW_RULE (eventfd);
	ALLOW_RULE (eventfd2);
	ALLOW_RULE (pipe);
	ALLOW_RULE (pipe2);
	ALLOW_RULE (epoll_create);
	ALLOW_RULE (epoll_create1);
	ALLOW_RULE (epoll_ctl);
	ALLOW_RULE (epoll_wait);
	ALLOW_RULE (epoll_pwait);
	ALLOW_RULE (epoll_pwait2);
	/* System */
	ALLOW_RULE (uname);
	ALLOW_RULE (sysinfo);
	ALLOW_RULE (prctl);
	ALLOW_RULE (getrandom);
	ALLOW_RULE (clock_gettime);
	ALLOW_RULE (clock_gettime64);
	ALLOW_RULE (clock_getres);
	ALLOW_RULE (gettimeofday);
	ALLOW_RULE (timerfd_create);
	ERROR_RULE (ioctl, EBADF);
	/* Descriptors */
	CUSTOM_RULE (close, SCMP_ACT_ALLOW, SCMP_CMP (0, SCMP_CMP_GT, STDERR_FILENO));
	CUSTOM_RULE (dup2, SCMP_ACT_ALLOW, SCMP_CMP (1, SCMP_CMP_GT, STDERR_FILENO));
	CUSTOM_RULE (dup3, SCMP_ACT_ALLOW, SCMP_CMP (1, SCMP_CMP_GT, STDERR_FILENO));
	ALLOW_RULE (read);
	ALLOW_RULE (lseek);
	ALLOW_RULE (_llseek);
	ALLOW_RULE (fadvise64);
	ALLOW_RULE (fadvise64_64);
	ALLOW_RULE (arm_fadvise64_64);
	ALLOW_RULE (write);
	ALLOW_RULE (writev);
	ALLOW_RULE (dup);
	/* Peer to peer D-Bus communication */
	ERROR_RULE (connect, EACCES);
	ALLOW_RULE (send);
	ALLOW_RULE (sendto);
	ALLOW_RULE (sendmsg);
	ALLOW_RULE (recv);
	ALLOW_RULE (recvmsg);
	ALLOW_RULE (recvfrom);
	ALLOW_RULE (getsockname);
	ALLOW_RULE (getpeername);
	ALLOW_RULE (getsockopt);
	ERROR_RULE (socket, EPERM);
	ERROR_RULE (setsockopt, EBADF);
	ERROR_RULE (bind, EACCES);
	/* File monitors */
	ALLOW_RULE (name_to_handle_at);
	ERROR_RULE (inotify_init1, EINVAL);
	ERROR_RULE (inotify_init, EINVAL);

	/* Allow tgkill on self, for abort() and friends */
	CUSTOM_RULE (tgkill, SCMP_ACT_ALLOW, SCMP_CMP(0, SCMP_CMP_EQ, getpid()));

	/* Allow prlimit64, only if no new limits are being set */
	CUSTOM_RULE (prlimit64, SCMP_ACT_ALLOW, SCMP_CMP(2, SCMP_CMP_EQ, 0));

	/* Special requirements for socketpair, only on AF_UNIX */
	CUSTOM_RULE (socketpair, SCMP_ACT_ALLOW, SCMP_CMP(0, SCMP_CMP_EQ, AF_UNIX));

#ifdef HAVE_BTRFS_IOCTL
	/* Special requirements for btrfs, allowed for BTRFS_IOC_INO_LOOKUP */
	CUSTOM_RULE (ioctl, SCMP_ACT_ALLOW, SCMP_CMP(1, SCMP_CMP_EQ, BTRFS_IOC_INO_LOOKUP));
#endif

	/* Special requirements for open/openat, allow O_RDONLY calls,
         * but fail if write permissions are requested.
	 */
	CUSTOM_RULE (open, SCMP_ACT_ALLOW,
	             SCMP_CMP (1, SCMP_CMP_MASKED_EQ,
	                       O_WRONLY | O_RDWR | O_APPEND | O_CREAT | O_TRUNC | O_EXCL, 0));
	CUSTOM_RULE (open, SCMP_ACT_ERRNO (EACCES), SCMP_CMP(1, SCMP_CMP_MASKED_EQ, O_WRONLY, O_WRONLY));
	CUSTOM_RULE (open, SCMP_ACT_ERRNO (EACCES), SCMP_CMP(1, SCMP_CMP_MASKED_EQ, O_RDWR, O_RDWR));

	CUSTOM_RULE (openat, SCMP_ACT_ALLOW,
	             SCMP_CMP (2, SCMP_CMP_MASKED_EQ,
	                       O_WRONLY | O_RDWR | O_APPEND | O_CREAT | O_TRUNC | O_EXCL, 0));
	CUSTOM_RULE (openat, SCMP_ACT_ERRNO (EACCES), SCMP_CMP(2, SCMP_CMP_MASKED_EQ, O_WRONLY, O_WRONLY));
	CUSTOM_RULE (openat, SCMP_ACT_ERRNO (EACCES), SCMP_CMP(2, SCMP_CMP_MASKED_EQ, O_RDWR, O_RDWR));

	/* Syscalls may differ between libcs */
#if !defined(__GLIBC__)
	ALLOW_RULE (readv);
#else
	ALLOW_RULE (pread64);
#endif

	g_debug ("Loading seccomp rules.");

	if (seccomp_load (ctx) >= 0) {
		/* Any seccomp filters loaded into the kernel are not affected. */
		seccomp_release (ctx);
		return TRUE;
	}

out:
	g_critical ("Failed to load seccomp rule for syscall '%s'", current_syscall);
	seccomp_release (ctx);
	return FALSE;
}

#else /* HAVE_LIBSECCOMP */

gboolean
tracker_seccomp_init (void)
{
	g_warning ("No seccomp support compiled-in.");
	return TRUE;
}

#endif /* HAVE_LIBSECCOMP */
