/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
#ifndef _OS_LOG_SIMPLE_PRIVATE_H_
#define _OS_LOG_SIMPLE_PRIVATE_H_

#include <os/log_simple_private_impl.h>
/**
 * These constants have the same value as os_log_type from os/log.h
 */
#define OS_LOG_SIMPLE_TYPE_DEFAULT 0x00
#define OS_LOG_SIMPLE_TYPE_INFO 0x01
#define OS_LOG_SIMPLE_TYPE_DEBUG 0x02
#define OS_LOG_SIMPLE_TYPE_ERROR 0x10

#define os_log_simple(fmt, ...)\
		os_log_simple_with_type(OS_LOG_SIMPLE_TYPE_DEFAULT, (fmt), ##__VA_ARGS__)

#define os_log_simple_error(fmt, ...)\
		os_log_simple_with_type(OS_LOG_SIMPLE_TYPE_ERROR, (fmt), ##__VA_ARGS__)

#define os_log_simple_with_type(type, fmt, ...)\
		os_log_simple_with_subsystem((type), NULL, (fmt), ##__VA_ARGS__)

#define os_log_simple_with_subsystem(type, subsystem, fmt, ...)\
		__os_log_simple_impl((type), (subsystem), (fmt), ##__VA_ARGS__)

#define os_log_simple_available() (&_os_log_simple != 0)

#endif /* _OS_LOG_SIMPLE_PRIVATE_H_ */
