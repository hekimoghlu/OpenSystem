/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#ifndef PROP_H
#define PROP_H 1
#include <Availability.h>

/* The following ifdef block is the standard way of creating macros
 * which make exporting from a DLL simpler. All files within this DLL
 * are compiled with the LIBSASL_EXPORTS symbol defined on the command
 * line. this symbol should not be defined on any project that uses
 * this DLL. This way any other project whose source files include
 * this file see LIBSASL_API functions as being imported from a DLL,
 * wheras this DLL sees symbols defined with this macro as being
 * exported.  */
/* Under Unix, life is simpler: we just need to mark library functions
 * as extern.  (Technically, we don't even have to do that.) */
#ifdef WIN32
# ifdef LIBSASL_EXPORTS
#  define LIBSASL_API  extern __declspec(dllexport)
# else /* LIBSASL_EXPORTS */
#  define LIBSASL_API  extern __declspec(dllimport)
# endif /* LIBSASL_EXPORTS */
#else /* WIN32 */
# define LIBSASL_API extern
#endif /* WIN32 */

/* Same as above, but used during a variable declaration. */
#ifdef WIN32
# ifdef LIBSASL_EXPORTS
#  define LIBSASL_VAR  extern __declspec(dllexport)
# else /* LIBSASL_EXPORTS */
#  define LIBSASL_VAR  extern __declspec(dllimport)
# endif /* LIBSASL_EXPORTS */
#else /* WIN32 */
# define LIBSASL_VAR extern
#endif /* WIN32 */

/* the resulting structure for property values
 */
struct propval {
    const char *name;	 /* name of property; NULL = end of list */
                         /* same pointer used in request will be used here */
    const char **values; /* list of strings, values == NULL if property not
			  * found, *values == NULL if property found with
			  * no values */
    unsigned nvalues;    /* total number of value strings */
    unsigned valsize;	 /* total size in characters of all value strings */
};

/*
 * private internal structure
 */
#define PROP_DEFAULT 4		/* default number of propvals to assume */
struct propctx;

#ifdef __cplusplus
extern "C" {
#endif

/* create a property context
 *  estimate -- an estimate of the storage needed for requests & responses
 *              0 will use module default
 * returns a new property context on success and NULL on any error
 */
LIBSASL_API struct propctx *prop_new(unsigned estimate) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* create new propctx which duplicates the contents of an existing propctx
 * returns SASL_OK on success
 * possible other return values include: SASL_NOMEM, SASL_BADPARAM
 */
LIBSASL_API int prop_dup(struct propctx *src_ctx, struct propctx **dst_ctx) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* Add property names to request
 *  ctx       -- context from prop_new()
 *  names     -- list of property names; must persist until context freed
 *               or requests cleared (This extends to other contexts that
 *               are dup'ed from this one, and their children, etc)
 *
 * NOTE: may clear values from context as side-effect
 * returns SASL_OK on success
 * possible other return values include: SASL_NOMEM, SASL_BADPARAM
 */
LIBSASL_API int prop_request(struct propctx *ctx, const char **names) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* return array of struct propval from the context
 *  return value persists until next call to
 *   prop_request, prop_clear or prop_dispose on context
 *
 *  returns NULL on error
 */
LIBSASL_API const struct propval *prop_get(struct propctx *ctx) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* Fill in an array of struct propval based on a list of property names
 *  return value persists until next call to
 *   prop_request, prop_clear or prop_dispose on context
 *  returns number of matching properties which were found (values != NULL)
 *  if a name requested here was never requested by a prop_request, then
 *  the name field of the associated vals entry will be set to NULL
 *
 * The vals array MUST be atleast as long as the names array.
 *
 * returns # of matching properties on success
 * possible other return values include: SASL_BADPARAM
 */
LIBSASL_API int prop_getnames(struct propctx *ctx, const char **names,
		  struct propval *vals) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* clear values and optionally requests from property context
 *  ctx      -- property context
 *  requests -- 0 = don't clear requests, 1 = clear requests
 */
LIBSASL_API void prop_clear(struct propctx *ctx, int requests) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* erase the value of a property
 */
LIBSASL_API void prop_erase(struct propctx *ctx, const char *name) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* dispose of property context
 *  ctx      -- is disposed and set to NULL; noop if ctx or *ctx is NULL
 */
LIBSASL_API void prop_dispose(struct propctx **ctx) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);


/****fetcher interfaces****/

/* format the requested property names into a string
 *  ctx    -- context from prop_new()/prop_request()
 *  sep    -- separator between property names (unused if none requested)
 *  seplen -- length of separator, if < 0 then strlen(sep) will be used
 *  outbuf -- output buffer
 *  outmax -- maximum length of output buffer including NUL terminator
 *  outlen -- set to length of output string excluding NUL terminator
 * returns SASL_OK on success
 * returns SASL_BADPARAM or amount of additional space needed on failure
 */
LIBSASL_API int prop_format(struct propctx *ctx, const char *sep, int seplen,
		char *outbuf, unsigned outmax, unsigned *outlen) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* add a property value to the context
 *  ctx    -- context from prop_new()/prop_request()
 *  name   -- name of property to which value will be added
 *            if NULL, add to the same name as previous prop_set/setvals call
 *  value  -- a value for the property; will be copied into context
 *            if NULL, remove existing values
 *  vallen -- length of value, if <= 0 then strlen(value) will be used
 * returns SASL_OK on success
 * possible error return values include: SASL_BADPARAM, SASL_NOMEM
 */
LIBSASL_API int prop_set(struct propctx *ctx, const char *name,
	     const char *value, int vallen) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

/* set the values for a property
 *  ctx    -- context from prop_new()/prop_request()
 *  name   -- name of property to which value will be added
 *            if NULL, add to the same name as previous prop_set/setvals call
 *  values -- array of values, ending in NULL.  Each value is a NUL terminated
 *            string
 * returns SASL_OK on success
 * possible error return values include: SASL_BADPARAM, SASL_NOMEM
 */
LIBSASL_API int prop_setvals(struct propctx *ctx, const char *name,
		 const char **values) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0,__MAC_10_11,__IPHONE_NA,__IPHONE_NA);

#ifdef __cplusplus
}
#endif

#endif /* PROP_H */
