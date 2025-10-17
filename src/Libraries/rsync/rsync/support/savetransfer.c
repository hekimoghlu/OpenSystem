/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#include "../rsync.h"

#define TIMEOUT_SECONDS 30

void run_program(char **command);

char buf[4096];
int save_data_from_program = 0;

int
main(int argc, char *argv[])
{
    int fd_file, len;
    struct timeval tv;
    fd_set fds;

    argv++;
    if (--argc && argv[0][0] == '-') {
	if (argv[0][1] == 'o')
	    save_data_from_program = 1;
	else if (argv[0][1] == 'i')
	    save_data_from_program = 0;
	else {
	    fprintf(stderr, "Unknown option: %s\n", argv[0]);
	    exit(1);
	}
	argv++;
	argc--;
    }
    if (argc < 2) {
	fprintf(stderr, "Usage: savetransfer [-i|-o] OUTPUT_FILE PROGRAM [ARGS...]\n");
	fprintf(stderr, "-i  Save the input going to PROGRAM to the OUTPUT_FILE\n");
	fprintf(stderr, "-o  Save the output coming from PROGRAM to the OUTPUT_FILE\n");
	exit(1);
    }
    if ((fd_file = open(*argv, O_WRONLY|O_TRUNC|O_CREAT|O_BINARY, 0644)) < 0) {
	fprintf(stderr, "Unable to write to `%s': %s\n", *argv, strerror(errno));
	exit(1);
    }
    set_blocking(fd_file);

    signal(SIGPIPE, SIG_IGN);

    run_program(argv + 1);

#if defined HAVE_SETMODE && O_BINARY
    setmode(STDIN_FILENO, O_BINARY);
    setmode(STDOUT_FILENO, O_BINARY);
#endif
    set_nonblocking(STDIN_FILENO);
    set_blocking(STDOUT_FILENO);

    while (1) {
	FD_ZERO(&fds);
	FD_SET(STDIN_FILENO, &fds);
	tv.tv_sec = TIMEOUT_SECONDS;
	tv.tv_usec = 0;
	if (!select(STDIN_FILENO+1, &fds, NULL, NULL, &tv))
	    break;
	if (!FD_ISSET(STDIN_FILENO, &fds))
	    break;
	if ((len = read(STDIN_FILENO, buf, sizeof buf)) <= 0)
	    break;
	if (write(STDOUT_FILENO, buf, len) != len) {
	    fprintf(stderr, "Failed to write data to stdout: %s\n", strerror(errno));
	    exit(1);
	}
	if (write(fd_file, buf, len) != len) {
	    fprintf(stderr, "Failed to write data to fd_file: %s\n", strerror(errno));
	    exit(1);
	}
    }
    return 0;
}

void
run_program(char **command)
{
    int pipe_fds[2], ret;
    pid_t pid;

    if (pipe(pipe_fds) < 0) {
	fprintf(stderr, "pipe failed: %s\n", strerror(errno));
	exit(1);
    }

    if ((pid = fork()) < 0) {
	fprintf(stderr, "fork failed: %s\n", strerror(errno));
	exit(1);
    }

    if (pid == 0) {
	if (save_data_from_program)
	    ret = dup2(pipe_fds[1], STDOUT_FILENO);
	else
	    ret = dup2(pipe_fds[0], STDIN_FILENO);
	if (ret < 0) {
	    fprintf(stderr, "Failed to dup (in child): %s\n", strerror(errno));
	    exit(1);
	}
	close(pipe_fds[0]);
	close(pipe_fds[1]);
	set_blocking(STDIN_FILENO);
	set_blocking(STDOUT_FILENO);
	execvp(command[0], command);
	fprintf(stderr, "Failed to exec %s: %s\n", command[0], strerror(errno));
	exit(1);
    }

    if (save_data_from_program)
	ret = dup2(pipe_fds[0], STDIN_FILENO);
    else
	ret = dup2(pipe_fds[1], STDOUT_FILENO);
    if (ret < 0) {
	fprintf(stderr, "Failed to dup (in parent): %s\n", strerror(errno));
	exit(1);
    }
    close(pipe_fds[0]);
    close(pipe_fds[1]);
}

void
set_nonblocking(int fd)
{
    int val;

    if ((val = fcntl(fd, F_GETFL, 0)) == -1)
	return;
    if (!(val & NONBLOCK_FLAG)) {
	val |= NONBLOCK_FLAG;
	fcntl(fd, F_SETFL, val);
    }
}

void
set_blocking(int fd)
{
    int val;

    if ((val = fcntl(fd, F_GETFL, 0)) < 0)
	return;
    if (val & NONBLOCK_FLAG) {
	val &= ~NONBLOCK_FLAG;
	fcntl(fd, F_SETFL, val);
    }
}
