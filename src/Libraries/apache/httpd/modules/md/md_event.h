/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#ifndef md_event_h
#define md_event_h

struct md_job_t;
struct md_result_t;

typedef apr_status_t md_event_cb(const char *event, 
                                 const char *mdomain, 
                                 void *baton,
                                 struct md_job_t *job, 
                                 struct md_result_t *result, 
                                 apr_pool_t *p);

void md_event_init(apr_pool_t *p); 

void md_event_subscribe(md_event_cb *cb, void *baton); 

apr_status_t md_event_raise(const char *event, 
                            const char *mdomain, 
                            struct md_job_t *job, 
                            struct md_result_t *result, 
                            apr_pool_t *p);

void md_event_holler(const char *event, 
                     const char *mdomain, 
                     struct md_job_t *job, 
                     struct md_result_t *result, 
                     apr_pool_t *p);

#endif /* md_event_h */
