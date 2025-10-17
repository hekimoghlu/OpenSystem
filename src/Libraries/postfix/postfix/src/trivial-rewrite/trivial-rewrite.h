/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include <vstring.h>
#include <vstream.h>

 /*
  * Global library.
  */
#include <tok822.h>
#include <maps.h>

 /*
  * Connection management.
  */
int     server_flags;

 /*
  * rewrite.c
  */
typedef struct {
    const char *origin_name;		/* name of variable */
    char  **origin;			/* default origin */
    const char *domain_name;		/* name of variable */
    char  **domain;			/* default domain */
} RWR_CONTEXT;

#define REW_PARAM_VALUE(x) (*(x))	/* make it easy to do it right */

extern void rewrite_init(void);
extern int rewrite_proto(VSTREAM *);
extern void rewrite_addr(RWR_CONTEXT *, char *, VSTRING *);
extern void rewrite_tree(RWR_CONTEXT *, TOK822 *);
extern RWR_CONTEXT local_context;
extern RWR_CONTEXT inval_context;

 /*
  * resolve.c
  */
typedef struct {
    const char *local_transport_name;	/* name of variable */
    char  **local_transport;		/* local transport:nexthop */
    const char *virt_transport_name;	/* name of variable */
    char  **virt_transport;		/* virtual mailbox transport:nexthop */
    const char *relay_transport_name;	/* name of variable */
    char  **relay_transport;		/* relay transport:nexthop */
    const char *def_transport_name;	/* name of variable */
    char  **def_transport;		/* default transport:nexthop */
    const char *snd_def_xp_maps_name;	/* name of variable */
    char  **snd_def_xp_maps;		/* maptype:mapname */
    MAPS   *snd_def_xp_info;		/* handle */
    const char *relayhost_name;		/* name of variable */
    char  **relayhost;			/* for relay and default transport */
    const char *snd_relay_maps_name;	/* name of variable */
    char  **snd_relay_maps;		/* maptype:mapname */
    MAPS   *snd_relay_info;		/* handle */
    const char *transport_maps_name;	/* name of variable */
    char  **transport_maps;		/* maptype:mapname */
    struct TRANSPORT_INFO *transport_info;	/* handle */
} RES_CONTEXT;

#define RES_PARAM_VALUE(x) (*(x))	/* make it easy to do it right */

extern void resolve_init(void);
extern int resolve_proto(RES_CONTEXT *, VSTREAM *);
extern int resolve_class(const char *);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/
