/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
/*
 * Kudos to Mark Huizer who originally suggested this on freebsd-current
 */

#include <sys/types.h>
#include <sys/uio.h>

#include <sys/socket.h>
#include <netinet/in.h>

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_LONG_REQUEST_LINE      1
#define TEST_LONG_REQUEST_FIELDS    2
#define TEST_LONG_REQUEST_FIELDSIZE 3
#define TEST_LONG_REQUEST_BODY      4

void
usage(void)
{
    fprintf(stderr,
      "usage: test_limits [-t (r|n|h|b)] [-a address] [-p port] [-n num]\n");
    exit(1);
}

int
main(int argc, char *argv[])
{
    struct sockaddr_in sin;
    struct hostent *he;
    FILE *f;
    int o, sd;

    /* default parameters */
    char *addr = "localhost";
    int port = 80;
    int num = 1000;
    int testtype = TEST_LONG_REQUEST_FIELDS;

    /* get options */
    while ((o = getopt(argc, argv, "t:a:p:n:")) != EOF)
        switch (o) {
        case 't':
            if (*optarg == 'r')
                testtype = TEST_LONG_REQUEST_LINE;
            else if (*optarg == 'n')
                testtype = TEST_LONG_REQUEST_FIELDS;
            else if (*optarg == 'h')
                testtype = TEST_LONG_REQUEST_FIELDSIZE;
            else if (*optarg == 'b')
                testtype = TEST_LONG_REQUEST_BODY;
            break;
        case 'a':
            addr = optarg;
            break;
        case 'p':
            port = atoi(optarg);
            break;
        case 'n':
            num = atoi(optarg);
            break;
        default:
            usage();
        }

    if (argc != optind)
        usage();

    /* connect */
    if ((he = gethostbyname(addr)) == NULL) {
        perror("gethostbyname");
        exit(1);
    }
    memset(&sin, sizeof(sin));
    memcpy((char *)&sin.sin_addr, he->h_addr, he->h_length);
    sin.sin_family = he->h_addrtype;
    sin.sin_port = htons(port);

    if ((sd = socket(sin.sin_family, SOCK_STREAM, IPPROTO_TCP)) == -1) {
        perror("socket");
        exit(1);
    }

    if (connect(sd, (struct sockaddr *)&sin, sizeof(sin)) == -1) {
        perror("connect");
        exit(1);
    }

    if ((f = fdopen(sd, "r+")) == NULL) {
        perror("fdopen");
        exit(1);
    }

    /* attack! */
    fprintf(stderr, "Testing like a plague of locusts on %s\n", addr);

    if (testtype == TEST_LONG_REQUEST_LINE) {
        fprintf(f, "GET ");
        while (num-- && !ferror(f)) {
            fprintf(f, "/123456789");
            fflush(f);
        }
        fprintf(f, " HTTP/1.0\r\n\r\n");
    }
    else {
        fprintf(f, "GET /fred/foo HTTP/1.0\r\n");

        if (testtype == TEST_LONG_REQUEST_FIELDSIZE) {
            while (num-- && !ferror(f)) {
                fprintf(f, "User-Agent: sioux");
                fflush(f);
            }
            fprintf(f, "\r\n");
        }
        else if (testtype == TEST_LONG_REQUEST_FIELDS) {
            while (num-- && !ferror(f))
                fprintf(f, "User-Agent: sioux\r\n");
            fprintf(f, "\r\n");
        }
        else if (testtype == TEST_LONG_REQUEST_BODY) {
            fprintf(f, "User-Agent: sioux\r\n");
            fprintf(f, "Content-Length: 33554433\r\n");
            fprintf(f, "\r\n");
            while (num-- && !ferror(f))
                fprintf(f, "User-Agent: sioux\r\n");
        }
        else {
            fprintf(f, "\r\n");
        }
    }
    fflush(f);

    {
        apr_ssize_t len;
        char buff[512];

        while ((len = read(sd, buff, 512)) > 0)
            len = write(1, buff, len);
    }
    if (ferror(f)) {
        perror("fprintf");
        exit(1);
    }

    fclose(f);
    exit(0);
}
