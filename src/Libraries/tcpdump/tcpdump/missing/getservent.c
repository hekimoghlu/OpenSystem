/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <netdissect-stdinc.h>
#include <getservent.h>

static FILE *servf = NULL;
static char line[BUFSIZ+1];
static struct servent serv;
static char *serv_aliases[MAXALIASES];
int _serv_stayopen;
const char *etc_path(const char *file);

/*
* Check if <file> exists in the current directory and, if so, return it.
* Else return either "%SYSTEMROOT%\System32\drivers\etc\<file>"
* or $PREFIX/etc/<file>.
* "<file>" is aka __PATH_SERVICES (aka "services" on Windows and
* "/etc/services" on other platforms that would need this).
*/
const char *etc_path(const char *file)
{
    const char *env = getenv(__PATH_SYSROOT);
    static char path[_MAX_PATH];

    /* see if "<file>" exists locally or whether __PATH_SYSROOT is valid */
    if (fopen(file, "r") || !env)
        return (file);
    else
#ifdef _WIN32
    snprintf(path, sizeof(path), "%s%s%s", env, __PATH_ETC_INET, file);
#else
    snprintf(path, sizeof(path), "%s%s", env, file);
#endif
    return (path);
}

void
setservent(int f)
{
    if (servf == NULL)
        servf = fopen(etc_path(__PATH_SERVICES), "r");
    else
        rewind(servf);
    _serv_stayopen |= f;
}

void
endservent(void)
{
    if (servf) {
        fclose(servf);
        servf = NULL;
    }
    _serv_stayopen = 0;
}

struct servent *
getservent(void)
{
    char *p;
    char *cp, **q;

    if (servf == NULL && (servf = fopen(etc_path(__PATH_SERVICES), "r")) == NULL)
        return (NULL);

again:
    if ((p = fgets(line, BUFSIZ, servf)) == NULL)
        return (NULL);
    if (*p == '#')
        goto again;
    cp = strpbrk(p, "#\n");
    if (cp == NULL)
        goto again;
    *cp = '\0';
    serv.s_name = p;
    p = strpbrk(p, " \t");
    if (p == NULL)
        goto again;
    *p++ = '\0';
    while (*p == ' ' || *p == '\t')
        p++;
    cp = strpbrk(p, ",/");
    if (cp == NULL)
        goto again;
    *cp++ = '\0';
    serv.s_port = htons((u_short)atoi(p));
    serv.s_proto = cp;
    q = serv.s_aliases = serv_aliases;
    cp = strpbrk(cp, " \t");
    if (cp != NULL)
        *cp++ = '\0';
    while (cp && *cp) {
        if (*cp == ' ' || *cp == '\t') {
            cp++;
            continue;
        }
        if (q < &serv_aliases[MAXALIASES - 1])
            *q++ = cp;
        cp = strpbrk(cp, " \t");
        if (cp != NULL)
            *cp++ = '\0';
    }
    *q = NULL;
    return (&serv);
}
