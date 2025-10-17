/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#ifndef _SECURITY_AUTH_CONNECTION_H_
#define _SECURITY_AUTH_CONNECTION_H_

#if defined(__cplusplus)
extern "C" {
#endif

AUTH_WARN_RESULT AUTH_MALLOC AUTH_NONNULL_ALL AUTH_RETURNS_RETAINED
connection_t connection_create(process_t);

AUTH_NONNULL_ALL
pid_t connection_get_pid(connection_t);
    
AUTH_NONNULL_ALL
process_t connection_get_process(connection_t);
    
AUTH_NONNULL_ALL
dispatch_queue_t connection_get_dispatch_queue(connection_t);

AUTH_NONNULL1
void connection_set_engine(connection_t, engine_t);

AUTH_NONNULL_ALL
void connection_destroy_agents(connection_t);

AUTH_NONNULL_ALL
bool connection_get_syslog_warn(connection_t);

AUTH_NONNULL_ALL
void connection_set_syslog_warn(connection_t);
    
#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_CONNECTION_H_ */
