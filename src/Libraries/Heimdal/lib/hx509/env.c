/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#include "hx_locl.h"

/**
 * @page page_env Hx509 enviroment functions
 *
 * See the library functions here: @ref hx509_env
 */

/**
 * Add a new key/value pair to the hx509_env.
 *
 * @param context A hx509 context.
 * @param env enviroment to add the enviroment variable too.
 * @param key key to add
 * @param value value to add
 *
 * @return An hx509 error code, see hx509_get_error_string().
 *
 * @ingroup hx509_env
 */

int
hx509_env_add(hx509_context context, hx509_env *env,
	      const char *key, const char *value)
{
    hx509_env n;

    n = malloc(sizeof(*n));
    if (n == NULL) {
	hx509_set_error_string(context, 0, ENOMEM, "out of memory");
	return ENOMEM;
    }

    n->type = env_string;
    n->next = NULL;
    n->name = strdup(key);
    if (n->name == NULL) {
	free(n);
	return ENOMEM;
    }
    n->u.string = strdup(value);
    if (n->u.string == NULL) {
	free(n->name);
	free(n);
	return ENOMEM;
    }

    /* add to tail */
    if (*env) {
	hx509_env e = *env;
	while (e->next)
	    e = e->next;
	e->next = n;
    } else
	*env = n;

    return 0;
}

/**
 * Add a new key/binding pair to the hx509_env.
 *
 * @param context A hx509 context.
 * @param env enviroment to add the enviroment variable too.
 * @param key key to add
 * @param list binding list to add
 *
 * @return An hx509 error code, see hx509_get_error_string().
 *
 * @ingroup hx509_env
 */

int
hx509_env_add_binding(hx509_context context, hx509_env *env,
		      const char *key, hx509_env list)
{
    hx509_env n;

    n = malloc(sizeof(*n));
    if (n == NULL) {
	hx509_set_error_string(context, 0, ENOMEM, "out of memory");
	return ENOMEM;
    }

    n->type = env_list;
    n->next = NULL;
    n->name = strdup(key);
    if (n->name == NULL) {
	free(n);
	return ENOMEM;
    }
    n->u.list = list;

    /* add to tail */
    if (*env) {
	hx509_env e = *env;
	while (e->next)
	    e = e->next;
	e->next = n;
    } else
	*env = n;

    return 0;
}


/**
 * Search the hx509_env for a length based key.
 *
 * @param context A hx509 context.
 * @param env enviroment to add the enviroment variable too.
 * @param key key to search for.
 * @param len length of key.
 *
 * @return the value if the key is found, NULL otherwise.
 *
 * @ingroup hx509_env
 */

const char *
hx509_env_lfind(hx509_context context, hx509_env env,
		const char *key, size_t len)
{
    while(env) {
	if (strncmp(key, env->name ,len) == 0
	    && env->name[len] == '\0' && env->type == env_string)
	    return env->u.string;
	env = env->next;
    }
    return NULL;
}

/**
 * Search the hx509_env for a key.
 *
 * @param context A hx509 context.
 * @param env enviroment to add the enviroment variable too.
 * @param key key to search for.
 *
 * @return the value if the key is found, NULL otherwise.
 *
 * @ingroup hx509_env
 */

const char *
hx509_env_find(hx509_context context, hx509_env env, const char *key)
{
    while(env) {
	if (strcmp(key, env->name) == 0 && env->type == env_string)
	    return env->u.string;
	env = env->next;
    }
    return NULL;
}

/**
 * Search the hx509_env for a binding.
 *
 * @param context A hx509 context.
 * @param env enviroment to add the enviroment variable too.
 * @param key key to search for.
 *
 * @return the binding if the key is found, NULL if not found.
 *
 * @ingroup hx509_env
 */

hx509_env
hx509_env_find_binding(hx509_context context,
		       hx509_env env,
		       const char *key)
{
    while(env) {
	if (strcmp(key, env->name) == 0 && env->type == env_list)
	    return env->u.list;
	env = env->next;
    }
    return NULL;
}

static void
env_free(hx509_env b)
{
    while(b) {
	hx509_env next = b->next;

	if (b->type == env_string)
	    free(b->u.string);
	else if (b->type == env_list)
	    env_free(b->u.list);

	free(b->name);
	free(b);
	b = next;
    }
}

/**
 * Free an hx509_env enviroment context.
 *
 * @param env the enviroment to free.
 *
 * @ingroup hx509_env
 */

void
hx509_env_free(hx509_env *env)
{
    if (*env)
	env_free(*env);
    *env = NULL;
}
