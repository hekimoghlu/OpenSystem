/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
/* $Id: xfrout.h,v 1.12 2007/06/19 23:46:59 tbox Exp $ */

#ifndef NAMED_XFROUT_H
#define NAMED_XFROUT_H 1

/*****
 ***** Module Info
 *****/

/*! \file 
 * \brief
 * Outgoing zone transfers (AXFR + IXFR).
 */

/***
 *** Functions
 ***/

void
ns_xfr_start(ns_client_t *client, dns_rdatatype_t xfrtype);

#endif /* NAMED_XFROUT_H */
