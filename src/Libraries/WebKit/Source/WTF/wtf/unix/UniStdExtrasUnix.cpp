/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#include "config.h"
#include <wtf/UniStdExtras.h>

#include <fcntl.h>

namespace WTF {

bool setCloseOnExec(int fileDescriptor)
{
    int returnValue = -1;
    do {
        int flags = fcntl(fileDescriptor, F_GETFD);
        if (flags != -1)
            returnValue = fcntl(fileDescriptor, F_SETFD, flags | FD_CLOEXEC);
    } while (returnValue == -1 && errno == EINTR);

    return returnValue != -1;
}

bool unsetCloseOnExec(int fileDescriptor)
{
    int returnValue = -1;
    do {
        int flags = fcntl(fileDescriptor, F_GETFD);
        if (flags != -1)
            returnValue = fcntl(fileDescriptor, F_SETFD, flags & ~FD_CLOEXEC);
    } while (returnValue == -1 && errno == EINTR);

    return returnValue != -1;
}

int dupCloseOnExec(int fileDescriptor)
{
    int duplicatedFileDescriptor = -1;
#ifdef F_DUPFD_CLOEXEC
    while ((duplicatedFileDescriptor = fcntl(fileDescriptor, F_DUPFD_CLOEXEC, 0)) == -1 && errno == EINTR) { }
    if (duplicatedFileDescriptor != -1)
        return duplicatedFileDescriptor;

#endif

    while ((duplicatedFileDescriptor = dup(fileDescriptor)) == -1 && errno == EINTR) { }
    if (duplicatedFileDescriptor == -1)
        return -1;

    if (!setCloseOnExec(duplicatedFileDescriptor)) {
        closeWithRetry(duplicatedFileDescriptor);
        return -1;
    }

    return duplicatedFileDescriptor;
}

bool setNonBlock(int fileDescriptor)
{
    int returnValue = -1;

    int flags = fcntl(fileDescriptor, F_GETFL, 0);
    while ((returnValue = fcntl(fileDescriptor, F_SETFL, flags | O_NONBLOCK)) == -1 && errno == EINTR) { }

    return returnValue != -1;
}

} // namespace WTF
