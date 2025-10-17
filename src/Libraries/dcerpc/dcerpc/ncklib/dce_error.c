/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
/*
 *  OSF DCE Version 1.0
 */
/*
**
**  NAME
**
**      dce_error.c
**
**  FACILITY:
**
**      Distributed Computing Environment (DCE)
**
**  ABSTRACT:
**
**  Error status management routines.
**
**
*/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#ifdef HAVE_NL_TYPES_H
#include <nl_types.h>       /* public types for NLS (I18N) routines */
#else
#warning Message catalog support disabled
#endif /* HAVE_NL_TYPES_H */

#include <dce/dce_error.h>

#define FACILITY_CODE_MASK          0xF0000000
#define FACILITY_CODE_SHIFT         28

#define COMPONENT_CODE_MASK         0x0FFFF000
#define COMPONENT_CODE_SHIFT        12

#define STATUS_CODE_MASK            0x00000FFF
#define STATUS_CODE_SHIFT           0

#define NO_MESSAGE                  "THIS IS NOT A MESSAGE"

/*
 * The system-dependant location for the catalog files is defined in sysconf.h
 */

#ifndef RPC_DEFAULT_NLSPATH
#define RPC_DEFAULT_NLSPATH "/usr/lib/nls/msg/en_US.ISO8859-1/%s.cat"
/* #error Define RPC_DEFAULT_NLSPATH in your sysconf.h file. */
#endif

#ifndef RPC_NLS_FORMAT
#define RPC_NLS_FORMAT "%s.cat"
#endif

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif


/*
**++
**
**  ROUTINE NAME:       dce_error_inq_text
**
**  SCOPE:              PUBLIC - declared in dce_error.h
**
**  DESCRIPTION:
**
**  Returns a text string in a user provided buffer associated with a given
**  error status code. In the case of errors a text string will also be
**  returned indicating the nature of the error.
**
**  INPUTS:
**
**      status_to_convert   A DCE error status code to be converted to
**                          text form.
**
**  INPUTS/OUTPUTS:         None.
**
**  OUTPUTS:
**
**      error_text          A user provided buffer to hold the text
**                          equivalent of status_to_convert or
**                          a message indicating what error occurred.
**
**
**      status              The result of the operation. One of:
**                           0  -  success
**                          -1  -  failure
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/

static void dce_get_msg(
	unsigned long   status_to_convert,
	char			*error_text,
	size_t			error_text_len,
	char			*fname,
	char			*cname,
	int             *status)
{
    unsigned short  facility_code;
    unsigned short  component_code;
    unsigned short  status_code;
#ifdef HAVE_NL_TYPES_H
    nl_catd     catd;
#endif
    char        component_name[4];
    const char  *facility_name;
    char        filename_prefix[7];
    char        nls_filename[MAXPATHLEN];
    char        alt_filename[MAXPATHLEN];
    char        *message;
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyz_0123456789-+@";
    static const char *facility_names[] = {
        "dce",
        "dfs"
    };

    /*
     * set up output status for future error returns
     */
    if (status != NULL)
    {
        *status = -1;
    }

    /*
     * check for ok input status
     */
    if (status_to_convert == 0)
    {
        if (status != NULL)
        {
            *status = 0;
        }
        strlcpy ((char *)error_text, "successful completion", error_text_len);
        return;
    }

    /*
     * extract the component, facility and status codes
     */
    facility_code = (status_to_convert & FACILITY_CODE_MASK)
        >> FACILITY_CODE_SHIFT;

    component_code = (status_to_convert & COMPONENT_CODE_MASK)
        >> COMPONENT_CODE_SHIFT;

    status_code = (status_to_convert & STATUS_CODE_MASK)
        >> STATUS_CODE_SHIFT;

    /*
     * see if this is a recognized facility
     */
    if (facility_code == 0 || facility_code > sizeof (facility_names) / sizeof (char *))
    {
        sprintf ((char *) error_text, "status %08lx (unknown facility)", status_to_convert);
        return;
    }

    facility_name = facility_names[facility_code - 1];

    /*
     * Convert component name from RAD-50 component code.  (Mapping is:
     * 0 => 'a', ..., 25 => 'z', 26 => '{', 27 => '0', ..., 36 => '9'.)
     */

    component_name[3] = 0;
    component_name[2] = alphabet[component_code % 40];
    component_code /= 40;
    component_name[1] = alphabet[component_code % 40];
    component_name[0] = alphabet[component_code / 40];

    if (fname != NULL)
        sprintf ((char*) fname, "%3s", facility_name);
    if (cname != NULL)
        sprintf ((char*) cname, "%3s", component_name);

    sprintf ((char*) filename_prefix, "%3s%3s", facility_name, component_name);

#if defined(CATALOG_DIR)
    sprintf ((char*) nls_filename,
	    CATALOG_DIR "/" RPC_NLS_FORMAT, filename_prefix);
#else
    sprintf ((char*) nls_filename, RPC_NLS_FORMAT, filename_prefix);
#endif

    /*
     * Open the message file
     */
#ifdef HAVE_NL_TYPES_H
    catd = (nl_catd) catopen (nls_filename, 0);
    if (catd == (nl_catd) -1)
    {
        /*
         * If we did not succeed in opening message file using NLSPATH,
         * try to open the message file in a well-known default area
         */

        sprintf (alt_filename,
                 RPC_DEFAULT_NLSPATH,
                 filename_prefix);
        catd = (nl_catd) catopen (alt_filename, 0);

        if (catd == (nl_catd) -1)
        {
            sprintf ((char *) error_text, "status %08lx", status_to_convert);
            return;
        }
    }

    /*
     * try to get the specified message from the file
     */
    message = (char *) catgets (catd, 1, status_code, NO_MESSAGE);

    /*
     * if everything went well, return the resulting message
     */
    if (strcmp (message, NO_MESSAGE) != 0)
    {
        sprintf ((char *) error_text, "%s", message);
        if (status != NULL)
        {
            *status = 0;
        }
    }
    else
    {
        sprintf ((char *) error_text, "status %08lx", status_to_convert);
    }

    catclose (catd);
#else
    sprintf ((char *) error_text, "status %08lx", status_to_convert);
#endif
}
void dce_error_inq_text (
unsigned long           status_to_convert,
dce_error_string_t      error_text,
int                     *status
)
{
    char        cname[4];
    char        fname[4];

    /*
     * check for ok input status
     */
    if (status_to_convert == 0)
    {
        if (status != NULL)
        {
            *status = 0;
        }
        strlcpy ((char *)error_text, "successful completion", dce_c_error_string_len);
        return;
    }

    dce_get_msg (status_to_convert, (char *)error_text, dce_c_error_string_len, fname, cname, status);
    strlcat ((char*) error_text, " (", dce_c_error_string_len);
    strlcat ((char*) error_text, fname, dce_c_error_string_len);
    strlcat ((char*) error_text, " / ", dce_c_error_string_len);
    strlcat ((char*) error_text, cname, dce_c_error_string_len);
    strlcat ((char*) error_text, ")", dce_c_error_string_len);
}

#if 0
/* unused functions */
int dce_fprintf(FILE *f, unsigned long index, ...)
{
    va_list ap;
    int st;
    int i;
    char format[1024];

    dce_get_msg(index, format, sizeof (format), NULL, NULL, &st);
    if (st != 0) return EOF;

    va_start(ap, index);
    i = vfprintf(f, format, ap);
    va_end(ap);
    return i;
}
int dce_printf(unsigned long index, ...)
{
    va_list ap;
    int st;
    int i;
    char format[1024];

    dce_get_msg(index, format, sizeof (format), NULL, NULL, &st);
    if (st != 0) return EOF;

    va_start(ap, index);
    i = vfprintf(stdout, format, ap);
    va_end(ap);
    return i;
}
#endif

#ifdef BUILD_STCODE
main(int argc, char **argv)
{
    long code;
    int i;
    int _;
    dce_error_string_t message;

    if (argc <= 1) {
        printf("Usage:  stcode {0x<hex status code> | <decimal status code>}\n");
        exit(1);
    }

    for (i=1; i < argc; i++) {
        if(strncmp(argv[i], "0x", 2) == 0)
                sscanf(argv[i], "%x", &code);
        else
                sscanf(argv[i], "%d", &code);
        dce_error_inq_text(code, message, &_);
        printf("%d (decimal), %x (hex): %s\n", code, code, message);
    }
}
#endif
