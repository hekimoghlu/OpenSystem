/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
/* $Id: lwaddr.h,v 1.8 2007/06/19 23:46:59 tbox Exp $ */

/*! \file */

#include <lwres/lwres.h>
#include <lwres/net.h>

isc_result_t
lwaddr_netaddr_fromlwresaddr(isc_netaddr_t *na, lwres_addr_t *la);

isc_result_t
lwaddr_sockaddr_fromlwresaddr(isc_sockaddr_t *sa, lwres_addr_t *la,
			      in_port_t port);

isc_result_t
lwaddr_lwresaddr_fromnetaddr(lwres_addr_t *la, isc_netaddr_t *na);

isc_result_t
lwaddr_lwresaddr_fromsockaddr(lwres_addr_t *la, isc_sockaddr_t *sa);
