/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
//  android.c
//  Socket
//
//  Created by Alsey Coleman Miller on 7/5/25.
//

#ifdef __ANDROID__

#include "CSystemAndroid.h"
#include <sys/socket.h>
#include <sys/sysinfo.h>
#include <sys/timerfd.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>

extern int android_fcntl(int fd, int cmd)
{
    return fcntl(fd, cmd);
}

extern int android_fcntl_value(int fd, int cmd, int value)
{
    return fcntl(fd, cmd, value);
}

extern int android_fcntl_ptr(int fd, int cmd, void* ptr)
{
    return fcntl(fd, cmd, ptr);
}

extern int android_ioctl(int fd, unsigned long op)
{
    return ioctl(fd, op);
}

extern int android_ioctl_value(int fd, unsigned long op, int value)
{
    return ioctl(fd, op, value);
}

extern int android_ioctl_ptr(int fd, unsigned long op, void* ptr)
{
    return ioctl(fd, op, ptr);
}

#endif
