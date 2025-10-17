/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#ifndef _SPRAY_H_RPCGEN
#define _SPRAY_H_RPCGEN

#define RPCGEN_VERSION	199506

#include <rpc/rpc.h>

#ifndef _RPCSVC_SPRAY_H_
#define _RPCSVC_SPRAY_H_

#define SPRAYOVERHEAD 86
#define SPRAYMAX 8845

struct spraytimeval {
	u_int sec;
	u_int usec;
};
typedef struct spraytimeval spraytimeval;
#ifdef __cplusplus
extern "C" bool_t xdr_spraytimeval(XDR *, spraytimeval*);
#elif __STDC__
extern  bool_t xdr_spraytimeval(XDR *, spraytimeval*);
#else /* Old Style C */
bool_t xdr_spraytimeval();
#endif /* Old Style C */


struct spraycumul {
	u_int counter;
	spraytimeval clock;
};
typedef struct spraycumul spraycumul;
#ifdef __cplusplus
extern "C" bool_t xdr_spraycumul(XDR *, spraycumul*);
#elif __STDC__
extern  bool_t xdr_spraycumul(XDR *, spraycumul*);
#else /* Old Style C */
bool_t xdr_spraycumul();
#endif /* Old Style C */


typedef struct {
	u_int sprayarr_len;
	char *sprayarr_val;
} sprayarr;
#ifdef __cplusplus
extern "C" bool_t xdr_sprayarr(XDR *, sprayarr*);
#elif __STDC__
extern  bool_t xdr_sprayarr(XDR *, sprayarr*);
#else /* Old Style C */
bool_t xdr_sprayarr();
#endif /* Old Style C */


#endif /* _RPCSVC_SPRAY_H_ */

#define SPRAYPROG ((rpc_uint)100012)
#define SPRAYVERS ((rpc_uint)1)

#ifdef __cplusplus
#define SPRAYPROC_SPRAY ((rpc_uint)1)
extern "C" void * sprayproc_spray_1(sprayarr *, CLIENT *);
extern "C" void * sprayproc_spray_1_svc(sprayarr *, struct svc_req *);
#define SPRAYPROC_GET ((rpc_uint)2)
extern "C" spraycumul * sprayproc_get_1(void *, CLIENT *);
extern "C" spraycumul * sprayproc_get_1_svc(void *, struct svc_req *);
#define SPRAYPROC_CLEAR ((rpc_uint)3)
extern "C" void * sprayproc_clear_1(void *, CLIENT *);
extern "C" void * sprayproc_clear_1_svc(void *, struct svc_req *);

#elif __STDC__
#define SPRAYPROC_SPRAY ((rpc_uint)1)
extern  void * sprayproc_spray_1(sprayarr *, CLIENT *);
extern  void * sprayproc_spray_1_svc(sprayarr *, struct svc_req *);
#define SPRAYPROC_GET ((rpc_uint)2)
extern  spraycumul * sprayproc_get_1(void *, CLIENT *);
extern  spraycumul * sprayproc_get_1_svc(void *, struct svc_req *);
#define SPRAYPROC_CLEAR ((rpc_uint)3)
extern  void * sprayproc_clear_1(void *, CLIENT *);
extern  void * sprayproc_clear_1_svc(void *, struct svc_req *);

#else /* Old Style C */
#define SPRAYPROC_SPRAY ((rpc_uint)1)
extern  void * sprayproc_spray_1();
extern  void * sprayproc_spray_1_svc();
#define SPRAYPROC_GET ((rpc_uint)2)
extern  spraycumul * sprayproc_get_1();
extern  spraycumul * sprayproc_get_1_svc();
#define SPRAYPROC_CLEAR ((rpc_uint)3)
extern  void * sprayproc_clear_1();
extern  void * sprayproc_clear_1_svc();
#endif /* Old Style C */

#endif /* !_SPRAY_H_RPCGEN */
