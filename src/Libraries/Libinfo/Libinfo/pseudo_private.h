/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#if (defined(DS_AVAILABLE))
int				_ds_running(void);
#endif
mach_port_t		_getaddrinfo_interface_async_call(const char *nodename, const char *servname, const struct addrinfo *hints, const char *interface, getaddrinfo_async_callback callback, void *context);
mach_port_t		_getnameinfo_interface_async_call(const struct sockaddr *sa, size_t len, int flags, const char *interface, getnameinfo_async_callback callback, void *context);
#if (defined(DS_AVAILABLE))
void			_si_disable_opendirectory(void);
#endif
extern uint32_t	gL1CacheEnabled;
int32_t			getgroupcount(const char *name, gid_t basegid);
int32_t			getgrouplist_2(const char *name, gid_t basegid, gid_t **groups);
