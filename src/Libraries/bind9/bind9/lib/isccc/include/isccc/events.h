/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
/* $Id: events.h,v 1.10 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_EVENTS_H
#define ISCCC_EVENTS_H 1

/*! \file isccc/events.h */

#include <isc/eventclass.h>

/*%
 * Registry of ISCCC event numbers.
 */

#define ISCCC_EVENT_CCMSG			(ISC_EVENTCLASS_ISCCC + 0)

#define ISCCC_EVENT_FIRSTEVENT			(ISC_EVENTCLASS_ISCCC + 0)
#define ISCCC_EVENT_LASTEVENT			(ISC_EVENTCLASS_ISCCC + 65535)

#endif /* ISCCC_EVENTS_H */
