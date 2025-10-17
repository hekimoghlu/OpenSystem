/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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

//
//  Copyright 2017 Apple. All rights reserved.
//

#include "AtomicFile.h"
#include <err.h>

#if 0
static void
fill_disk(const char *path)
{
    int fd = ::open(path, O_CREAT|O_RDWR, 0600);
    if (fd < 0)
        errx(1, "failed to create fill file");

    uint8 buffer[1024] = {};
    ::memset(reinterpret_cast<void *>(buffer), 0x77, sizeof(buffer));

    for (unsigned count = 0; count < 1000; count++) {
        if (::write(fd, buffer, sizeof(buffer)) != sizeof(buffer)) {
            warn("write fill file failed");
            break;
        }
    }
    if (close(fd) < 0)
        warn("close fill file failed");
}
#endif

int
main(int argc, char **argv)
{
    int fail = 0;

    if (argc != 2)
        errx(1, "argc != 2");

    try {
        AtomicFile file(argv[1]);

        RefPointer<AtomicTempFile> temp = file.write();

        unsigned count = 0;
        uint8 buffer[1024] = {};
        ::memset(reinterpret_cast<void *>(buffer), 0xff, sizeof(buffer));

        for (count = 0; count < 1000; count++) {
            temp->write(AtomicFile::FromEnd, 0, buffer, sizeof(buffer));
        }

        temp->commit();
        temp = NULL;
    } catch (...) {
        fail = 1;
    }
    if (fail)
        errx(1, "failed to create new file");
    return 0;
}



