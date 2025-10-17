/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
#include "curl_setup.h"

#ifndef CURL_DISABLE_DICT

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NET_IF_H
#include <net/if.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#elif defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include "urldata.h"
#include <curl/curl.h>
#include "transfer.h"
#include "sendf.h"
#include "escape.h"
#include "progress.h"
#include "dict.h"
#include "curl_printf.h"
#include "strcase.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/*
 * Forward declarations.
 */

static CURLcode dict_do(struct Curl_easy *data, bool *done);

/*
 * DICT protocol handler.
 */

const struct Curl_handler Curl_handler_dict = {
  "DICT",                               /* scheme */
  ZERO_NULL,                            /* setup_connection */
  dict_do,                              /* do_it */
  ZERO_NULL,                            /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  ZERO_NULL,                            /* disconnect */
  ZERO_NULL,                            /* write_resp */
  ZERO_NULL,                            /* connection_check */
  ZERO_NULL,                            /* attach connection */
  PORT_DICT,                            /* defport */
  CURLPROTO_DICT,                       /* protocol */
  CURLPROTO_DICT,                       /* family */
  PROTOPT_NONE | PROTOPT_NOURLQUERY     /* flags */
};

#define DYN_DICT_WORD 10000
static char *unescape_word(const char *input)
{
  struct dynbuf out;
  const char *ptr;
  CURLcode result = CURLE_OK;
  Curl_dyn_init(&out, DYN_DICT_WORD);

  /* According to RFC2229 section 2.2, these letters need to be escaped with
     \[letter] */
  for(ptr = input; *ptr; ptr++) {
    char ch = *ptr;
    if((ch <= 32) || (ch == 127) ||
       (ch == '\'') || (ch == '\"') || (ch == '\\'))
      result = Curl_dyn_addn(&out, "\\", 1);
    if(!result)
      result = Curl_dyn_addn(&out, ptr, 1);
    if(result)
      return NULL;
  }
  return Curl_dyn_ptr(&out);
}

/* sendf() sends formatted data to the server */
static CURLcode sendf(struct Curl_easy *data,
                      const char *fmt, ...) CURL_PRINTF(2, 3);

static CURLcode sendf(struct Curl_easy *data, const char *fmt, ...)
{
  size_t bytes_written;
  size_t write_len;
  CURLcode result = CURLE_OK;
  char *s;
  char *sptr;
  va_list ap;
  va_start(ap, fmt);
  s = vaprintf(fmt, ap); /* returns an allocated string */
  va_end(ap);
  if(!s)
    return CURLE_OUT_OF_MEMORY; /* failure */

  bytes_written = 0;
  write_len = strlen(s);
  sptr = s;

  for(;;) {
    /* Write the buffer to the socket */
    result = Curl_xfer_send(data, sptr, write_len, &bytes_written);

    if(result)
      break;

    Curl_debug(data, CURLINFO_DATA_OUT, sptr, (size_t)bytes_written);

    if((size_t)bytes_written != write_len) {
      /* if not all was written at once, we must advance the pointer, decrease
         the size left and try again! */
      write_len -= bytes_written;
      sptr += bytes_written;
    }
    else
      break;
  }

  free(s); /* free the output string */

  return result;
}

static CURLcode dict_do(struct Curl_easy *data, bool *done)
{
  char *word;
  char *eword = NULL;
  char *ppath;
  char *database = NULL;
  char *strategy = NULL;
  char *nthdef = NULL; /* This is not part of the protocol, but required
                          by RFC 2229 */
  CURLcode result;

  char *path;

  *done = TRUE; /* unconditionally */

  /* url-decode path before further evaluation */
  result = Curl_urldecode(data->state.up.path, 0, &path, NULL, REJECT_CTRL);
  if(result)
    return result;

  if(strncasecompare(path, DICT_MATCH, sizeof(DICT_MATCH)-1) ||
     strncasecompare(path, DICT_MATCH2, sizeof(DICT_MATCH2)-1) ||
     strncasecompare(path, DICT_MATCH3, sizeof(DICT_MATCH3)-1)) {

    word = strchr(path, ':');
    if(word) {
      word++;
      database = strchr(word, ':');
      if(database) {
        *database++ = (char)0;
        strategy = strchr(database, ':');
        if(strategy) {
          *strategy++ = (char)0;
          nthdef = strchr(strategy, ':');
          if(nthdef) {
            *nthdef = (char)0;
          }
        }
      }
    }

    if(!word || (*word == (char)0)) {
      infof(data, "lookup word is missing");
      word = (char *)"default";
    }
    if(!database || (*database == (char)0)) {
      database = (char *)"!";
    }
    if(!strategy || (*strategy == (char)0)) {
      strategy = (char *)".";
    }

    eword = unescape_word(word);
    if(!eword) {
      result = CURLE_OUT_OF_MEMORY;
      goto error;
    }

    result = sendf(data,
                   "CLIENT " LIBCURL_NAME " " LIBCURL_VERSION "\r\n"
                   "MATCH "
                   "%s "    /* database */
                   "%s "    /* strategy */
                   "%s\r\n" /* word */
                   "QUIT\r\n",
                   database,
                   strategy,
                   eword);

    if(result) {
      failf(data, "Failed sending DICT request");
      goto error;
    }
    Curl_xfer_setup(data, FIRSTSOCKET, -1, FALSE, -1); /* no upload */
  }
  else if(strncasecompare(path, DICT_DEFINE, sizeof(DICT_DEFINE)-1) ||
          strncasecompare(path, DICT_DEFINE2, sizeof(DICT_DEFINE2)-1) ||
          strncasecompare(path, DICT_DEFINE3, sizeof(DICT_DEFINE3)-1)) {

    word = strchr(path, ':');
    if(word) {
      word++;
      database = strchr(word, ':');
      if(database) {
        *database++ = (char)0;
        nthdef = strchr(database, ':');
        if(nthdef) {
          *nthdef = (char)0;
        }
      }
    }

    if(!word || (*word == (char)0)) {
      infof(data, "lookup word is missing");
      word = (char *)"default";
    }
    if(!database || (*database == (char)0)) {
      database = (char *)"!";
    }

    eword = unescape_word(word);
    if(!eword) {
      result = CURLE_OUT_OF_MEMORY;
      goto error;
    }

    result = sendf(data,
                   "CLIENT " LIBCURL_NAME " " LIBCURL_VERSION "\r\n"
                   "DEFINE "
                   "%s "     /* database */
                   "%s\r\n"  /* word */
                   "QUIT\r\n",
                   database,
                   eword);

    if(result) {
      failf(data, "Failed sending DICT request");
      goto error;
    }
    Curl_xfer_setup(data, FIRSTSOCKET, -1, FALSE, -1);
  }
  else {

    ppath = strchr(path, '/');
    if(ppath) {
      int i;

      ppath++;
      for(i = 0; ppath[i]; i++) {
        if(ppath[i] == ':')
          ppath[i] = ' ';
      }
      result = sendf(data,
                     "CLIENT " LIBCURL_NAME " " LIBCURL_VERSION "\r\n"
                     "%s\r\n"
                     "QUIT\r\n", ppath);
      if(result) {
        failf(data, "Failed sending DICT request");
        goto error;
      }

      Curl_xfer_setup(data, FIRSTSOCKET, -1, FALSE, -1);
    }
  }

error:
  free(eword);
  free(path);
  return result;
}
#endif /* CURL_DISABLE_DICT */
