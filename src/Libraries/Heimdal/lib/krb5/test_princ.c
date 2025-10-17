/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#include "krb5_locl.h"
#include <err.h>

/*
 * Check that a closed cc still keeps it data and that it's no longer
 * there when it's destroyed.
 */

static void
test_princ(krb5_context context)
{
    const char *princ = "lha@SU.SE";
    const char *princ_short = "lha";
    const char *noquote;
    krb5_error_code ret;
    char *princ_unparsed;
    char *princ_reformed = NULL;
    const char *realm;

    krb5_principal p, p2;

    ret = krb5_parse_name(context, princ, &p);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_unparse_name(context, p, &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (strcmp(princ, princ_unparsed)) {
	krb5_errx(context, 1, "%s != %s", princ, princ_unparsed);
    }

    free(princ_unparsed);

    ret = krb5_unparse_name_flags(context, p,
				  KRB5_PRINCIPAL_UNPARSE_NO_REALM,
				  &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (strcmp(princ_short, princ_unparsed))
	krb5_errx(context, 1, "%s != %s", princ_short, princ_unparsed);
    free(princ_unparsed);

    realm = krb5_principal_get_realm(context, p);

    if (asprintf(&princ_reformed, "%s@%s", princ_short, realm) < 0 || princ_reformed == NULL)
	errx(1, "malloc");

    ret = krb5_parse_name(context, princ_reformed, &p2);
    free(princ_reformed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (!krb5_principal_compare(context, p, p2)) {
	krb5_errx(context, 1, "p != p2");
    }

    krb5_free_principal(context, p2);

    ret = krb5_set_default_realm(context, "SU.SE");
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_unparse_name_flags(context, p,
				  KRB5_PRINCIPAL_UNPARSE_SHORT,
				  &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (strcmp(princ_short, princ_unparsed))
	krb5_errx(context, 1, "'%s' != '%s'", princ_short, princ_unparsed);
    free(princ_unparsed);

    ret = krb5_parse_name(context, princ_short, &p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (!krb5_principal_compare(context, p, p2))
	krb5_errx(context, 1, "p != p2");
    krb5_free_principal(context, p2);

    ret = krb5_unparse_name(context, p, &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (strcmp(princ, princ_unparsed))
	krb5_errx(context, 1, "'%s' != '%s'", princ, princ_unparsed);
    free(princ_unparsed);

    ret = krb5_set_default_realm(context, "SAMBA.ORG");
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_parse_name(context, princ_short, &p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (krb5_principal_compare(context, p, p2))
	krb5_errx(context, 1, "p == p2");

    if (!krb5_principal_compare_any_realm(context, p, p2))
	krb5_errx(context, 1, "(ignoring realms) p != p2");

    ret = krb5_unparse_name(context, p2, &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (strcmp(princ, princ_unparsed) == 0)
	krb5_errx(context, 1, "%s == %s", princ, princ_unparsed);
    free(princ_unparsed);

    krb5_free_principal(context, p2);

    ret = krb5_parse_name(context, princ, &p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (!krb5_principal_compare(context, p, p2))
	krb5_errx(context, 1, "p != p2");

    ret = krb5_unparse_name(context, p2, &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (strcmp(princ, princ_unparsed))
	krb5_errx(context, 1, "'%s' != '%s'", princ, princ_unparsed);
    free(princ_unparsed);

    krb5_free_principal(context, p2);

    ret = krb5_unparse_name_flags(context, p,
				  KRB5_PRINCIPAL_UNPARSE_SHORT,
				  &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name_short");

    if (strcmp(princ, princ_unparsed) != 0)
	krb5_errx(context, 1, "'%s' != '%s'", princ, princ_unparsed);
    free(princ_unparsed);

    ret = krb5_unparse_name(context, p, &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name_short");

    if (strcmp(princ, princ_unparsed))
	krb5_errx(context, 1, "'%s' != '%s'", princ, princ_unparsed);
    free(princ_unparsed);

    ret = krb5_parse_name_flags(context, princ,
				KRB5_PRINCIPAL_PARSE_NO_REALM,
				&p2);
    if (!ret)
	krb5_err(context, 1, ret, "Should have failed to parse %s a "
		 "short name", princ);

    ret = krb5_parse_name_flags(context, princ_short,
				KRB5_PRINCIPAL_PARSE_NO_REALM,
				&p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_unparse_name_flags(context, p2,
				  KRB5_PRINCIPAL_UNPARSE_NO_REALM,
				  &princ_unparsed);
    krb5_free_principal(context, p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name_norealm");

    if (strcmp(princ_short, princ_unparsed))
	krb5_errx(context, 1, "'%s' != '%s'", princ_short, princ_unparsed);
    free(princ_unparsed);

    ret = krb5_parse_name_flags(context, princ_short,
				KRB5_PRINCIPAL_PARSE_REQUIRE_REALM,
				&p2);
    if (!ret)
	krb5_err(context, 1, ret, "Should have failed to parse %s "
		 "because it lacked a realm", princ_short);

    ret = krb5_parse_name_flags(context, princ,
				KRB5_PRINCIPAL_PARSE_REQUIRE_REALM,
				&p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    if (!krb5_principal_compare(context, p, p2))
	krb5_errx(context, 1, "p != p2");

    ret = krb5_unparse_name_flags(context, p2,
				  KRB5_PRINCIPAL_UNPARSE_NO_REALM,
				  &princ_unparsed);
    krb5_free_principal(context, p2);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name_norealm");

    if (strcmp(princ_short, princ_unparsed))
	krb5_errx(context, 1, "'%s' != '%s'", princ_short, princ_unparsed);
    free(princ_unparsed);

    krb5_free_principal(context, p);

    /* test quoting */

    princ = "test\\ principal@SU.SE";
    noquote = "test principal@SU.SE";

    ret = krb5_parse_name_flags(context, princ, 0, &p);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_unparse_name_flags(context, p, 0, &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name_flags");

    if (strcmp(princ, princ_unparsed))
	krb5_errx(context, 1, "q '%s' != '%s'", princ, princ_unparsed);
    free(princ_unparsed);

    ret = krb5_unparse_name_flags(context, p, KRB5_PRINCIPAL_UNPARSE_DISPLAY,
				  &princ_unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name_flags");

    if (strcmp(noquote, princ_unparsed))
	krb5_errx(context, 1, "nq '%s' != '%s'", noquote, princ_unparsed);
    free(princ_unparsed);

    krb5_free_principal(context, p);
}

static void
test_enterprise(krb5_context context)
{
    krb5_error_code ret;
    char *unparsed;
    krb5_principal p;

    ret = krb5_set_default_realm(context, "SAMBA.ORG");
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name");

    ret = krb5_parse_name_flags(context, "lha@su.se@WIN.SU.SE",
				KRB5_PRINCIPAL_PARSE_ENTERPRISE, &p);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name_flags");

    ret = krb5_unparse_name(context, p, &unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name");

    krb5_free_principal(context, p);

    if (strcmp(unparsed, "lha\\@su.se@WIN.SU.SE") != 0)
	krb5_errx(context, 1, "enterprise name failed 1");
    free(unparsed);

    /*
     *
     */

    ret = krb5_parse_name_flags(context, "lha\\@su.se@WIN.SU.SE",
				KRB5_PRINCIPAL_PARSE_ENTERPRISE, &p);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name_flags");

    ret = krb5_unparse_name(context, p, &unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name");

    krb5_free_principal(context, p);
    if (strcmp(unparsed, "lha\\@su.se\\@WIN.SU.SE@SAMBA.ORG") != 0)
	krb5_errx(context, 1, "enterprise name failed 2: %s", unparsed);
    free(unparsed);

    /*
     *
     */

    ret = krb5_parse_name_flags(context, "lha\\@su.se@WIN.SU.SE", 0, &p);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name_flags");

    ret = krb5_unparse_name(context, p, &unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name");

    krb5_free_principal(context, p);
    if (strcmp(unparsed, "lha\\@su.se@WIN.SU.SE") != 0)
	krb5_errx(context, 1, "enterprise name failed 3");
    free(unparsed);

    /*
     *
     */

    ret = krb5_parse_name_flags(context, "lha@su.se",
				KRB5_PRINCIPAL_PARSE_ENTERPRISE, &p);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_name_flags");

    ret = krb5_unparse_name(context, p, &unparsed);
    if (ret)
	krb5_err(context, 1, ret, "krb5_unparse_name");

    krb5_free_principal(context, p);
    if (strcmp(unparsed, "lha\\@su.se@SAMBA.ORG") != 0)
	krb5_errx(context, 1, "enterprise name failed 2: %s", unparsed);
    free(unparsed);
}


int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;

    setprogname(argv[0]);

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    test_princ(context);

    test_enterprise(context);

    krb5_free_context(context);

    return 0;
}
