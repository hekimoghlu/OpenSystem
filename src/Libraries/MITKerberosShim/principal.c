/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include "heim.h"
#include <string.h>

static void
map_mit_principal(struct comb_principal *p)
{
    unsigned long i;

    p->mit.magic = MIT_KV5M_PRINCIPAL;
    p->mit.type = p->heim->name.name_type;
    p->mit.realm.magic = MIT_KV5M_DATA;
    p->mit.realm.data = p->heim->realm;
    p->mit.realm.length = (unsigned int)strlen(p->heim->realm);
    p->mit.data = calloc(p->heim->name.name_string.len, sizeof(*p->mit.data));
    for (i = 0; i < p->heim->name.name_string.len; i++) {
	p->mit.data[i].magic = MIT_KV5M_DATA;
	p->mit.data[i].data = p->heim->name.name_string.val[i];
	p->mit.data[i].length = (unsigned int)strlen(p->heim->name.name_string.val[i]);
    }
    p->mit.length = p->heim->name.name_string.len;
}

mit_krb5_principal
mshim_hprinc2mprinc(krb5_context context, krb5_principal princ)
{
    struct comb_principal *p;
    p = calloc(1, sizeof(*p));
    heim_krb5_copy_principal(context, princ, &p->heim);
    map_mit_principal(p);
    return (mit_krb5_principal)p;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_parse_name(mit_krb5_context context, const char *str, mit_krb5_principal *principal)
{
    return krb5_parse_name_flags(context, str, 0, principal);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_parse_name_flags(mit_krb5_context context, const char *str, int flags, mit_krb5_principal *principal)
{
    struct comb_principal *p;
    krb5_error_code ret;
    
    LOG_ENTRY();
    
    p = calloc(1, sizeof(*p));
    ret = heim_krb5_parse_name_flags((krb5_context)context, str, flags, &p->heim);
    if (ret) {
        free(p);
        return ret;
    }
    map_mit_principal(p);
    *principal = (mit_krb5_principal)p;
    return 0;
}


mit_krb5_error_code KRB5_CALLCONV_C
krb5_build_principal_ext(mit_krb5_context context, mit_krb5_principal *principal, unsigned int rlen, const char *realm, ...)
{
    struct comb_principal *p;
    krb5_error_code ret;
    va_list ap;

    LOG_ENTRY();

    va_start(ap, realm);
    p = calloc(1, sizeof(*p));
    ret = heim_krb5_build_principal_va_ext((krb5_context)context, &p->heim, rlen, realm, ap);
    va_end(ap);
    if (ret) {
	free(p);
	return ret;
    }
    map_mit_principal(p);
    *principal = (mit_krb5_principal)p;
    return ret;
}

mit_krb5_error_code KRB5_CALLCONV_C
krb5_build_principal(mit_krb5_context context, mit_krb5_principal *principal, unsigned int rlen, const char *realm, ...)
{
    struct comb_principal *p;
    krb5_error_code ret;
    va_list ap;

    LOG_ENTRY();

    va_start(ap, realm);
    p = calloc(1, sizeof(*p));
    ret = heim_krb5_build_principal_va((krb5_context)context, &p->heim, rlen, realm, ap);
    va_end(ap);
    if (ret) {
	free(p);
	return ret;
    }
    map_mit_principal(p);
    *principal = (mit_krb5_principal)p;
    return ret;
}

mit_krb5_error_code KRB5_CALLCONV
krb5_unparse_name(mit_krb5_context context, mit_krb5_const_principal principal, char **str)
{
    struct comb_principal *p = (struct comb_principal *)principal;
    LOG_ENTRY();
    return heim_krb5_unparse_name((krb5_context)context, p->heim, str);
}

void KRB5_CALLCONV
krb5_free_unparsed_name(mit_krb5_context context, char *str)
{
    LOG_ENTRY();
    heim_krb5_xfree(str);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_copy_principal(mit_krb5_context context,
		    mit_krb5_const_principal from,
		    mit_krb5_principal *to)
{
    struct comb_principal *p = (struct comb_principal *)from;
    LOG_ENTRY();
    *to = mshim_hprinc2mprinc(HC(context), p->heim);
    return 0;
}

void KRB5_CALLCONV
krb5_free_principal(mit_krb5_context context, mit_krb5_principal principal)
{
    struct comb_principal *p = (struct comb_principal *)principal;
    LOG_ENTRY();
    if (p) {
	heim_krb5_free_principal(HC(context), p->heim);
	free(p->mit.data);
	free(p);
    }
}

void KRB5_CALLCONV
krb5_free_default_realm(mit_krb5_context context, char *str)
{
    LOG_ENTRY();
    free(str);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_sname_to_principal(mit_krb5_context context,
			const char *hostname, const char *service, 
			mit_krb5_int32 type,
			mit_krb5_principal *principal)
{
    krb5_error_code ret;
    krb5_principal p;

    LOG_ENTRY();

    *principal = NULL;

    ret = heim_krb5_sname_to_principal(HC(context), hostname, service, type, &p);
    if (ret)
	return ret;

    *principal = mshim_hprinc2mprinc(HC(context), p);
    heim_krb5_free_principal(HC(context), p);
    return 0;
}

mit_krb5_boolean KRB5_CALLCONV
krb5_principal_compare(mit_krb5_context context,
		       mit_krb5_const_principal p1,
		       mit_krb5_const_principal p2)
{
    struct comb_principal *c1 = (struct comb_principal *)p1;
    struct comb_principal *c2 = (struct comb_principal *)p2;

    return heim_krb5_principal_compare(HC(context), c1->heim, c2->heim);
}

mit_krb5_boolean KRB5_CALLCONV
krb5_realm_compare(mit_krb5_context context,
		   mit_krb5_const_principal p1,
		   mit_krb5_const_principal p2)
{
    struct comb_principal *c1 = (struct comb_principal *)p1;
    struct comb_principal *c2 = (struct comb_principal *)p2;

    return heim_krb5_realm_compare(HC(context), c1->heim, c2->heim);
}

mit_krb5_error_code KRB5_CALLCONV
krb5_get_realm_domain(mit_krb5_context, const char *, char **);


mit_krb5_error_code KRB5_CALLCONV
krb5_get_realm_domain(mit_krb5_context context, const char *realm, char **domain)
{
    const char *d;

    d = heim_krb5_config_get_string(HC(context), NULL, "realms", realm,
				    "default_realm", NULL);
    if (d == NULL) {
	*domain = NULL;
	return (-1429577726L); /* PROF_NO_SECTION */
    }
    *domain = strdup(d);
    return 0;
}
