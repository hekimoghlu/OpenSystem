/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#ifndef _TRUSTD_SPI_h
#define _TRUSTD_SPI_h

#include <stdio.h>
#include <xpc/private.h>
#include <xpc/xpc.h>
#include <CoreFoundation/CFURL.h>

// Don't call these functions unless you are trustd
extern struct trustd trustd_spi;

void trustd_init_server(void);
void trustd_init(CFURLRef home_dir);
void trustd_exit_clean(const char *reason);

#endif /* _TRUSTD_SPI_h */
