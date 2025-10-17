/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

#include <sys/un.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
// Utility that does the equivalent of python bind for unix domain socket
int main(int argc, char **argv) {
    char * s_path = argv[1];
    if (!s_path) return(1);
    struct sockaddr_un s;
    int fd;
    s.sun_family = AF_UNIX;
    strncpy(s.sun_path, (char *)s_path, sizeof(s.sun_path));
    fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    bind(fd, (struct sockaddr *) &s, sizeof(struct sockaddr_un));
    close(fd);
    return 0;
}

