/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#ifndef _LAK_H
#define _LAK_H

#include <ldap.h>
#include <lber.h>

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#define LAK_OK 0
#define LAK_FAIL -1
#define LAK_NOMEM -2
#define LAK_RETRY -3
#define LAK_NOT_GROUP_MEMBER -4
#define LAK_INVALID_PASSWORD -5
#define LAK_USER_NOT_FOUND -6
#define LAK_BIND_FAIL -7
#define LAK_CONNECT_FAIL -8

#define LAK_NOT_BOUND 1
#define LAK_BOUND 2

#define LAK_AUTH_METHOD_BIND 0
#define LAK_AUTH_METHOD_CUSTOM 1
#define LAK_AUTH_METHOD_FASTBIND 2

#define LAK_GROUP_MATCH_METHOD_ATTR 0
#define LAK_GROUP_MATCH_METHOD_FILTER 1

#define LAK_BUF_LEN 128
#define LAK_DN_LEN 512
#define LAK_PATH_LEN 1024
#define LAK_URL_LEN LAK_PATH_LEN

typedef struct lak_conf {
    char   path[LAK_PATH_LEN];
    char   servers[LAK_URL_LEN];
    char   bind_dn[LAK_DN_LEN];
    char   password[LAK_BUF_LEN];
    int    version;
    struct timeval timeout;
    int    size_limit;
    int    time_limit;
    int    deref;
    int    referrals;
    int    restart;
    int    scope;
    char   default_realm[LAK_BUF_LEN];
    char   search_base[LAK_DN_LEN];
    char   filter[LAK_DN_LEN];
    char   password_attr[LAK_BUF_LEN];
    char   group_dn[LAK_DN_LEN];
    char   group_attr[LAK_BUF_LEN];
    char   group_filter[LAK_DN_LEN];
    char   group_search_base[LAK_DN_LEN];
    int    group_scope;
    int    group_match_method;
    char   auth_method;
    int    use_sasl;
    char   id[LAK_BUF_LEN];
    char   authz_id[LAK_BUF_LEN];
    char   mech[LAK_BUF_LEN];
    char   realm[LAK_BUF_LEN];
    char   sasl_secprops[LAK_BUF_LEN];
    int    start_tls;
    int    tls_check_peer;
    char   tls_cacert_file[LAK_PATH_LEN];
    char   tls_cacert_dir[LAK_PATH_LEN];
    char   tls_ciphers[LAK_BUF_LEN];
    char   tls_cert[LAK_PATH_LEN];
    char   tls_key[LAK_PATH_LEN];
    int    debug;
} LAK_CONF;

typedef struct lak_user {
    char bind_dn[LAK_DN_LEN];
    char id[LAK_BUF_LEN];
    char authz_id[LAK_BUF_LEN];
    char mech[LAK_BUF_LEN];
    char realm[LAK_BUF_LEN];
    char password[LAK_BUF_LEN];
} LAK_USER;


typedef struct lak {
    LDAP     *ld;
    char      status;
    LAK_USER *user;
    LAK_CONF *conf;
} LAK;

typedef struct lak_result {
    char              *attribute;
    char              *value;
    size_t             len;
    struct lak_result *next;
} LAK_RESULT;

int lak_init(const char *, LAK **);
void lak_close(LAK *);
int lak_authenticate(LAK *, const char *, const char *, const char *, const char *);
int lak_retrieve(LAK *, const char *, const char *, const char *, const char **, LAK_RESULT **);
void lak_result_free(LAK_RESULT *);
char *lak_error(const int errno);

#endif  /* _LAK_H */
