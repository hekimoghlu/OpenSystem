/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <engine.h>

#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#ifndef RTLD_NOW
#define RTLD_NOW 0
#endif
#endif

struct hc_engine {
    int references;
    char *name;
    char *id;
    void (*destroy)(ENGINE *);
    const RSA_METHOD *rsa;
    const ECDSA_METHOD *ecdsa;
    const DH_METHOD *dh;
    const RAND_METHOD *rand;
};

ENGINE	*
ENGINE_new(void)
{
    ENGINE *engine;

    engine = calloc(1, sizeof(*engine));
    engine->references = 1;

    return engine;
}

int
ENGINE_free(ENGINE *engine)
{
    return ENGINE_finish(engine);
}

int
ENGINE_finish(ENGINE *engine)
{
    if (engine->references-- <= 0)
	abort();
    if (engine->references > 0)
	return 1;

    if (engine->name)
	free(engine->name);
    if (engine->id)
	free(engine->id);
    if(engine->destroy)
	(*engine->destroy)(engine);

    memset(engine, 0, sizeof(*engine));
    engine->references = -1;


    free(engine);
    return 1;
}

int
ENGINE_up_ref(ENGINE *engine)
{
    if (engine->references < 0)
	abort();
    engine->references++;
    return 1;
}

int
ENGINE_set_id(ENGINE *engine, const char *id)
{
    engine->id = strdup(id);
    return (engine->id == NULL) ? 0 : 1;
}

int
ENGINE_set_name(ENGINE *engine, const char *name)
{
    engine->name = strdup(name);
    return (engine->name == NULL) ? 0 : 1;
}

int
ENGINE_set_ECDSA(ENGINE *engine, const ECDSA_METHOD *method)
{
    engine->ecdsa = method;
    return 1;
}

int
ENGINE_set_RSA(ENGINE *engine, const RSA_METHOD *method)
{
    engine->rsa = method;
    return 1;
}

int
ENGINE_set_DH(ENGINE *engine, const DH_METHOD *method)
{
    engine->dh = method;
    return 1;
}

int
ENGINE_set_destroy_function(ENGINE *e, void (*destroy)(ENGINE *))
{
    e->destroy = destroy;
    return 1;
}

const char *
ENGINE_get_id(const ENGINE *engine)
{
    return engine->id;
}

const char *
ENGINE_get_name(const ENGINE *engine)
{
    return engine->name;
}

const RSA_METHOD *
ENGINE_get_RSA(const ENGINE *engine)
{
    return engine->rsa;
}

const ECDSA_METHOD *
ENGINE_get_ECDSA(const ENGINE *engine)
{
    return engine->ecdsa;
}

const DH_METHOD *
ENGINE_get_DH(const ENGINE *engine)
{
    return engine->dh;
}

const RAND_METHOD *
ENGINE_get_RAND(const ENGINE *engine)
{
    return engine->rand;
}

/*
 *
 */

#define SG_default_engine(type)			\
static ENGINE *type##_engine;			\
int						\
ENGINE_set_default_##type(ENGINE *engine)	\
{						\
    if (type##_engine)				\
	ENGINE_finish(type##_engine);		\
    type##_engine = engine;			\
    if (type##_engine)				\
	ENGINE_up_ref(type##_engine);		\
    return 1;					\
}						\
ENGINE *					\
ENGINE_get_default_##type(void)			\
{						\
    if (type##_engine)				\
	ENGINE_up_ref(type##_engine);		\
    return type##_engine;			\
}

SG_default_engine(RSA)
SG_default_engine(ECDSA)
SG_default_engine(DH)

#undef SG_default_engine

/*
 *
 */

static ENGINE **engines;
static unsigned int num_engines;

static int
add_engine(ENGINE *engine)
{
    ENGINE **d, *dup;

    dup = ENGINE_by_id(engine->id);
    if (dup)
	return 0;

    d = realloc(engines, (num_engines + 1) * sizeof(*engines));
    if (d == NULL)
	return 1;
    engines = d;
    engines[num_engines++] = engine;

    return 1;
}

void
ENGINE_load_builtin_engines(void)
{
    ENGINE *engine;
    int ret;

    engine = ENGINE_new();
    if (engine == NULL)
	return;

    ENGINE_set_id(engine, "builtin");
    ENGINE_set_name(engine,
		    "Heimdal crypto builtin engine version " PACKAGE_VERSION);
#ifdef HAVE_CDSA
    ENGINE_set_DH(engine, DH_cdsa_method());
    ENGINE_set_RSA(engine, RSA_cdsa_method());
#elif defined(HEIM_HC_SF)
    ENGINE_set_RSA(engine, RSA_sf_method());
    ENGINE_set_DH(engine, DH_sf_method());
#elif defined(HEIM_HC_LTM)
    ENGINE_set_RSA(engine, RSA_ltm_method());
    ENGINE_set_DH(engine, DH_ltm_method());
#else
    ENGINE_set_RSA(engine, RSA_tfm_method());
    ENGINE_set_DH(engine, DH_tfm_method());
#endif

    ret = add_engine(engine);
    if (ret != 1)
	ENGINE_finish(engine);

#ifdef USE_HCRYPTO_TFM
    /*
     * TFM
     */

    engine = ENGINE_new();
    if (engine == NULL)
	return;

    ENGINE_set_id(engine, "tfm");
    ENGINE_set_name(engine,
		    "Heimdal crypto tfm engine version " PACKAGE_VERSION);
    ENGINE_set_RSA(engine, RSA_tfm_method());
    ENGINE_set_DH(engine, DH_tfm_method());

    ret = add_engine(engine);
    if (ret != 1)
	ENGINE_finish(engine);
#endif /* USE_HCRYPTO_TFM */

#ifdef USE_HCRYPTO_LTM
    /*
     * ltm
     */

    engine = ENGINE_new();
    if (engine == NULL)
	return;

    ENGINE_set_id(engine, "ltm");
    ENGINE_set_name(engine,
		    "Heimdal crypto ltm engine version " PACKAGE_VERSION);
    ENGINE_set_RSA(engine, RSA_ltm_method());
    ENGINE_set_DH(engine, DH_ltm_method());

    ret = add_engine(engine);
    if (ret != 1)
	ENGINE_finish(engine);
#endif

#ifdef HAVE_GMP
    /*
     * gmp
     */

    engine = ENGINE_new();
    if (engine == NULL)
	return;

    ENGINE_set_id(engine, "gmp");
    ENGINE_set_name(engine,
		    "Heimdal crypto gmp engine version " PACKAGE_VERSION);
    ENGINE_set_RSA(engine, RSA_gmp_method());

    ret = add_engine(engine);
    if (ret != 1)
	ENGINE_finish(engine);
#endif
}

ENGINE *
ENGINE_by_dso(const char *path, const char *id)
{
#ifdef HAVE_DLOPEN
    ENGINE *engine;
    void *handle;
    int ret;

    engine = calloc(1, sizeof(*engine));
    if (engine == NULL)
	return NULL;

    handle = dlopen(path, RTLD_NOW);
    if (handle == NULL) {
	/* printf("error: %s\n", dlerror()); */
	free(engine);
	return NULL;
    }

    {
	unsigned long version;
	openssl_v_check v_check;

	v_check = (openssl_v_check)dlsym(handle, "v_check");
	if (v_check == NULL) {
	    dlclose(handle);
	    free(engine);
	    return NULL;
	}

	version = (*v_check)(OPENSSL_DYNAMIC_VERSION);
	if (version == 0) {
	    dlclose(handle);
	    free(engine);
	    return NULL;
	}
    }

    {
	openssl_bind_engine bind_engine;

	bind_engine = (openssl_bind_engine)dlsym(handle, "bind_engine");
	if (bind_engine == NULL) {
	    dlclose(handle);
	    free(engine);
	    return NULL;
	}

	ret = (*bind_engine)(engine, id, NULL); /* XXX fix third arg */
	if (ret != 1) {
	    dlclose(handle);
	    free(engine);
	    return NULL;
	}
    }

    ENGINE_up_ref(engine);

    ret = add_engine(engine);
    if (ret != 1) {
	dlclose(handle);
	ENGINE_finish(engine);
	return NULL;
    }

    return engine;
#else
    return NULL;
#endif
}

ENGINE *
ENGINE_by_id(const char *id)
{
    size_t i;

    for (i = 0; i < num_engines; i++) {
	if (strcmp(id, engines[i]->id) == 0) {
	    ENGINE_up_ref(engines[i]);
	    return engines[i];
	}
    }
    return NULL;
}

void
ENGINE_add_conf_module(void)
{
}
