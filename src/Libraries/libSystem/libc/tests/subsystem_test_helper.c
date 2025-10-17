/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <subsystem.h>
#include <sys/stat.h>
#include <sys/errno.h>
#include "subsystem_test.h"

#include <stdio.h>

/*
 * main expects 3 arguments:
 *   1: Requested behavior (see subsystem_test.h)
 *   2: Filepath to operate on
 *   3: String to write to the file (if applicable)
 */
int
main(int argc, char **argv)
{
    struct stat stat_buf;
    int syscall_return = 0;
    int retval = 1;
    ino_t inode;
    ino_t main_inode;

    if (argc != 4) {
        return retval;
    }
    
    char * behavior = argv[1];
    char * filepath = argv[2];
    char * write_string = argv[3];
    size_t write_string_len = strlen(write_string) + 1;
    
    if (!strcmp(behavior, HELPER_BEHAVIOR_OPEN_OVERFLOW)) {
        /*
         * Open with overflow case; expects a filepath longer
         * than PATH_MAX.
         */
        syscall_return = open_with_subsystem(filepath, O_RDWR);
        
        if (syscall_return < 0) {
            if (errno == ENAMETOOLONG) {
                retval = 0;
            }
        } else {
            close(syscall_return);
        }
    } else if (!strcmp(behavior, HELPER_BEHAVIOR_STAT_OVERFLOW)) {
        /*
         * Stat with overflow case; expects a filepath longer
         * than PATH_MAX.
         */
        syscall_return = stat_with_subsystem(filepath, &stat_buf);
        
        if ((syscall_return < 0) && (errno == ENAMETOOLONG)) {
            retval = 0;
        }
    }
    if (!strcmp(behavior, HELPER_BEHAVIOR_OPEN_O_CREAT)) {
        /*
         * Open with O_CREAT case; O_CREAT should never work
         * with open_with_subsystem.
         */
        syscall_return = open_with_subsystem(filepath, O_CREAT | O_RDWR);

        if ((syscall_return < 0) && (errno == EINVAL)) {
            retval = 0;
        } else {
            close(syscall_return);
        }
    } else if (!strcmp(behavior, HELPER_BEHAVIOR_STAT_NONE)) {
        /*
         * Stat when neither file is present.
         */
        syscall_return = stat_with_subsystem(filepath, &stat_buf);

        if (syscall_return) {
            retval = 0;
        }
    } else if (!strcmp(behavior, HELPER_BEHAVIOR_STAT_MAIN) ||
               !strcmp(behavior, HELPER_BEHAVIOR_STAT_NOT_MAIN)) {
        /*
         * Stat when at least one file is present.
         */
        syscall_return = stat_with_subsystem(filepath, &stat_buf);
        
        if (!syscall_return) {
            inode = stat_buf.st_ino;
            
            syscall_return = stat(filepath, &stat_buf);
            if (!syscall_return) {
                main_inode = stat_buf.st_ino;
                
                /* Compare inodes based on the requested behavior. */
                if (!strcmp(behavior, HELPER_BEHAVIOR_STAT_MAIN)) {
                    if (inode == main_inode) {
                        /* It was the main file. */
                        retval = 0;
                    }
                } else if (!strcmp(behavior, HELPER_BEHAVIOR_STAT_NOT_MAIN)) {
                    if (inode != main_inode) {
                        /* It was the subsystem file. */
                        retval = 0;
                    }
                }
            } else if (!strcmp(behavior, HELPER_BEHAVIOR_STAT_NOT_MAIN)) {
                /* If main doesn't exist, we found the subsystem file. */
                retval = 0;
            }
        }
    } else if (!strcmp(behavior, HELPER_BEHAVIOR_OPEN_AND_WRITE)) {
        /*
         * Open and write case; it is on the client to check that this
         * wrote to the expected file.
         */
        syscall_return = open_with_subsystem(filepath, O_RDWR | O_TRUNC);
        
        if (syscall_return >= 0) {
            write(syscall_return, write_string, write_string_len);
            close(syscall_return);
            retval = 0;
        }
    }

    return retval;
}
