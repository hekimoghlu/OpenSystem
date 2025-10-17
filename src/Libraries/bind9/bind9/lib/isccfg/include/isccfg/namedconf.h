/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
/* $Id: namedconf.h,v 1.18 2010/08/11 18:14:20 each Exp $ */

#ifndef ISCCFG_NAMEDCONF_H
#define ISCCFG_NAMEDCONF_H 1

/*! \file isccfg/namedconf.h
 * \brief
 * This module defines the named.conf, rndc.conf, and rndc.key grammars.
 */

#include <isccfg/cfg.h>

/*
 * Configuration object types.
 */
LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_namedconf;
/*%< A complete named.conf file. */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_bindkeys;
/*%< A bind.keys file. */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_newzones;
/*%< A new-zones file (for zones added by 'rndc addzone'). */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_addzoneconf;
/*%< A single zone passed via the addzone rndc command. */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_rndcconf;
/*%< A complete rndc.conf file. */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_rndckey;
/*%< A complete rndc.key file. */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_sessionkey;
/*%< A complete session.key file. */

LIBISCCFG_EXTERNAL_DATA extern cfg_type_t cfg_type_keyref;
/*%< A key reference, used as an ACL element */

#endif /* ISCCFG_NAMEDCONF_H */
