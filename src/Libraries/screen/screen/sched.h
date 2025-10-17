/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
struct event
{
  struct event *next;
  void (*handler) __P((struct event *, char *));
  char *data;
  int fd;
  int type;
  int pri;
  struct timeval timeout;
  int queued;		/* in evs queue */
  int active;		/* in fdset */
  int *condpos;		/* only active if condpos - condneg > 0 */
  int *condneg;
};

#define EV_TIMEOUT	0
#define EV_READ		1
#define EV_WRITE	2
#define EV_ALWAYS	3
