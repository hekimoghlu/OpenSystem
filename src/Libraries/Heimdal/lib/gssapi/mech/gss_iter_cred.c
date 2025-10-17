/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include "mech_locl.h"
#include <gssapi_spi.h>
#include <heim_threads.h>


struct _gss_iter {
    HEIMDAL_MUTEX mutex;
    unsigned int count;
    void *userctx;
    void (*iter)(void *, gss_const_OID, gss_cred_id_t);
};

static void
iter_deref(struct _gss_iter *ctx)
{
    HEIMDAL_MUTEX_lock(&ctx->mutex);
    if (--ctx->count == 0) {
	(ctx->iter)(ctx->userctx, NULL, NULL);
	HEIMDAL_MUTEX_unlock(&ctx->mutex);
	HEIMDAL_MUTEX_destroy(&ctx->mutex);
	free(ctx);
    } else
	HEIMDAL_MUTEX_unlock(&ctx->mutex);
}


static void
iterate(void *cctx, gss_OID mech, gss_cred_id_t cred)
{
    struct _gss_iter *ctx = cctx;
    if (cred) {
	struct _gss_mechanism_cred *mc;
	struct _gss_cred *c;

	c = _gss_mg_alloc_cred();
	if (!c)
	    return;

	mc = malloc(sizeof(struct _gss_mechanism_cred));
	if (!mc) {
	    free(c);
	    return;
	}

	mc->gmc_mech = __gss_get_mechanism(mech);
	mc->gmc_mech_oid = mech;
	mc->gmc_cred = cred;
	HEIM_SLIST_INSERT_HEAD(&c->gc_mc, mc, gmc_link);

	ctx->iter(ctx->userctx, mech, (gss_cred_id_t)c);

    } else {
	/*
	 * Now that we reach the end of this mechs credentials,
	 * release the context, only one ref per mech.
	 */
	iter_deref(ctx);
    }
}

/**
 * Iterate over all credentials
 *
 * @param min_stat set to minor status in case of an error
 * @param flags flags argument, no flags currently defined, pass in 0 (zero)
 * @param mech the mechanism type of credentials to iterate over, by passing in GSS_C_NO_OID, the function will iterate over all credentails
 * @param userctx user context passed to the useriter funcion
 * @param useriter function that will be called on each gss_cred_id_t, when NULL is passed the list is completed. Must free the credential with gss_release_cred().
 *
 * @ingroup gssapi
 */

OM_uint32 GSSAPI_LIB_FUNCTION
gss_iter_creds_f(OM_uint32 *__nonnull min_stat,
		 OM_uint32 flags,
		 __nullable gss_const_OID mech,
		 void * __nullable userctx,
		  void (*__nonnull useriter)(void *__nullable , __nullable gss_iter_OID, __nullable gss_cred_id_t))
{
    struct _gss_iter *ctx;
    gss_OID_set mechs;
    gssapi_mech_interface m;
    size_t i;

    if (useriter == NULL)
	return GSS_S_CALL_INACCESSIBLE_READ;
    
    _gss_load_mech();
    
    /*
     * First make sure that at least one of the requested
     * mechanisms is one that we support.
     */
    mechs = _gss_mech_oids;
    
    ctx = malloc(sizeof(struct _gss_iter));
    if (ctx == NULL) {
	if (min_stat)
	    *min_stat = ENOMEM;
	return GSS_S_FAILURE;
    }
    
    HEIMDAL_MUTEX_init(&ctx->mutex);
    ctx->count = 1;
    ctx->userctx = userctx;
    ctx->iter = useriter;
    
    for (i = 0; i < mechs->count; i++) {
	
	if (mech && !gss_oid_equal(mech, &mechs->elements[i]))
	    continue;
	
	m = __gss_get_mechanism(&mechs->elements[i]);
	if (!m)
	    continue;
	
	if (m->gm_iter_creds == NULL)
	    continue;
	
	HEIMDAL_MUTEX_lock(&ctx->mutex);
	ctx->count += 1;
	HEIMDAL_MUTEX_unlock(&ctx->mutex);
	
	m->gm_iter_creds(flags, ctx, iterate);
    }
    
    iter_deref(ctx);
    
    return GSS_S_COMPLETE;
}

#ifdef __BLOCKS__

#include <Block.h>

static void
useriter_block(void *ctx, gss_const_OID mech, gss_cred_id_t cred)
{
    void (^u)(gss_const_OID, gss_cred_id_t) = ctx;

    u(mech, cred);

    if (cred == NULL)
	Block_release(u);
	
}

/**
 * Iterate over all credentials
 *
 * @param min_stat set to minor status in case of an error
 * @param flags flags argument, no flags currently defined, pass in 0 (zero)
 * @param mech the mechanism type of credentials to iterate over, by passing in GSS_C_NO_OID, the function will iterate over all credentails
 * @param useriter block that will be called on each gss_cred_id_t, when NULL is passed the list is completed. Must free the credential with gss_release_cred().
 *
 * @ingroup gssapi
 */


OM_uint32 GSSAPI_LIB_FUNCTION
gss_iter_creds(OM_uint32 *__nonnull min_stat,
	       OM_uint32 flags,
	       __nullable gss_const_OID mech,
	       void (^__nonnull useriter)(__nullable gss_iter_OID, __nullable gss_cred_id_t))
{
    void (^u)(gss_const_OID, gss_cred_id_t) = (void (^)(gss_const_OID, gss_cred_id_t))Block_copy(useriter);

    return gss_iter_creds_f(min_stat, flags, mech, u, useriter_block);
}

#endif
