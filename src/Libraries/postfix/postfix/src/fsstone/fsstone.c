/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
/* System library. */

#include <sys_defs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

/* Utility library. */

#include <msg.h>
#include <msg_vstream.h>

/* Global directory. */

#include <mail_version.h>

/* rename_file - rename a file */

static void rename_file(int old, int new)
{
    char    new_path[BUFSIZ];
    char    old_path[BUFSIZ];

    sprintf(new_path, "%06d", new);
    sprintf(old_path, "%06d", old);
    if (rename(old_path, new_path))
	msg_fatal("rename %s to %s: %m", old_path, new_path);
}

/* make_file - create a little file and use it */

static void make_file(int seqno, int size)
{
    char    path[BUFSIZ];
    char    buf[1024];
    FILE   *fp;
    int     i;

    sprintf(path, "%06d", seqno);
    if ((fp = fopen(path, "w")) == 0)
	msg_fatal("open %s: %m", path);
    memset(buf, 'x', sizeof(buf));
    for (i = 0; i < size; i++)
	if (fwrite(buf, 1, sizeof(buf), fp) != sizeof(buf))
	    msg_fatal("fwrite: %m");
    if (fsync(fileno(fp)))
	msg_fatal("fsync: %m");
    if (fclose(fp))
	msg_fatal("fclose: %m");
    if ((fp = fopen(path, "r")) == 0)
	msg_fatal("open %s: %m", path);
    while (fgets(path, sizeof(path), fp))
	 /* void */ ;
    if (fclose(fp))
	msg_fatal("fclose: %m");
}

/* use_file - use existing file */

static void use_file(int seqno)
{
    char    path[BUFSIZ];
    FILE   *fp;
    int     i;

    sprintf(path, "%06d", seqno);
    if ((fp = fopen(path, "w")) == 0)
	msg_fatal("open %s: %m", path);
    for (i = 0; i < 400; i++)
	fprintf(fp, "hello");
    if (fsync(fileno(fp)))
	msg_fatal("fsync: %m");
    if (fclose(fp))
	msg_fatal("fclose: %m");
    if ((fp = fopen(path, "r+")) == 0)
	msg_fatal("open %s: %m", path);
    while (fgets(path, sizeof(path), fp))
	 /* void */ ;
    if (ftruncate(fileno(fp), (off_t) 0))
	msg_fatal("ftruncate: %m");;
    if (fclose(fp))
	msg_fatal("fclose: %m");
}

/* remove_file - delete specified file */

static void remove_file(int seq)
{
    char    path[BUFSIZ];

    sprintf(path, "%06d", seq);
    if (remove(path))
	msg_fatal("remove %s: %m", path);
}

/* remove_silent - delete specified file, silently */

static void remove_silent(int seq)
{
    char    path[BUFSIZ];

    sprintf(path, "%06d", seq);
    (void) remove(path);
}

/* usage - explain */

static void usage(char *myname)
{
    msg_fatal("usage: %s [-cr] [-s size] messages directory_entries", myname);
}

MAIL_VERSION_STAMP_DECLARE;

int     main(int argc, char **argv)
{
    int     op_count;
    int     max_file;
    struct timeval start, end;
    int     do_rename = 0;
    int     do_create = 0;
    int     seq;
    int     ch;
    int     size = 2;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    msg_vstream_init(argv[0], VSTREAM_ERR);
    while ((ch = GETOPT(argc, argv, "crs:")) != EOF) {
	switch (ch) {
	case 'c':
	    do_create++;
	    break;
	case 'r':
	    do_rename++;
	    break;
	case 's':
	    if ((size = atoi(optarg)) <= 0)
		usage(argv[0]);
	    break;
	default:
	    usage(argv[0]);
	}
    }

    if (argc - optind != 2 || (do_rename && !do_create))
	usage(argv[0]);
    if ((op_count = atoi(argv[optind])) <= 0)
	usage(argv[0]);
    if ((max_file = atoi(argv[optind + 1])) <= 0)
	usage(argv[0]);

    /*
     * Populate the directory with little files.
     */
    for (seq = 0; seq < max_file; seq++)
	make_file(seq, size);

    /*
     * Simulate arrival and delivery of mail messages.
     */
    GETTIMEOFDAY(&start);
    while (op_count > 0) {
	seq %= max_file;
	if (do_create) {
	    remove_file(seq);
	    make_file(seq, size);
	    if (do_rename) {
		rename_file(seq, seq + max_file);
		rename_file(seq + max_file, seq);
	    }
	} else {
	    use_file(seq);
	}
	seq++;
	op_count--;
    }
    GETTIMEOFDAY(&end);
    if (end.tv_usec < start.tv_usec) {
	end.tv_sec--;
	end.tv_usec += 1000000;
    }
    printf("elapsed time: %ld.%06ld\n",
	   (long) (end.tv_sec - start.tv_sec),
	   (long) (end.tv_usec - start.tv_usec));

    /*
     * Clean up directory fillers.
     */
    for (seq = 0; seq < max_file; seq++)
	remove_silent(seq);
    return (0);
}
