/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#ifndef _SECURITY_SECTRUSTLOGGINGSERVER_H_
#define _SECURITY_SECTRUSTLOGGINGSERVER_H_


void DisableLocalization(void);

/* interval in seconds since startup */
uint64_t TimeSinceSystemStartup(void);
/* nsec interval since trustd process launch */
uint64_t TimeSinceProcessLaunch(void);
/* nsec interval until trustd has been up for at least this many nsecs */
int64_t TimeUntilProcessUptime(int64_t uptime_nsecs);

#endif /* _SECURITY_SECTRUSTLOGGINGSERVER_H_ */
