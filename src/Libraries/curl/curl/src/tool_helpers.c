/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include "tool_setup.h"

#include "strcase.h"

#define ENABLE_CURLX_PRINTF
/* use our own printf() functions */
#include "curlx.h"

#include "tool_cfgable.h"
#include "tool_msgs.h"
#include "tool_getparam.h"
#include "tool_helpers.h"

#include "memdebug.h" /* keep this as LAST include */

/*
** Helper functions that are used from more than one source file.
*/

const char *param2text(int res)
{
  ParameterError error = (ParameterError)res;
  switch(error) {
  case PARAM_GOT_EXTRA_PARAMETER:
    return "had unsupported trailing garbage";
  case PARAM_OPTION_UNKNOWN:
    return "is unknown";
  case PARAM_OPTION_AMBIGUOUS:
    return "is ambiguous";
  case PARAM_REQUIRES_PARAMETER:
    return "requires parameter";
  case PARAM_BAD_USE:
    return "is badly used here";
  case PARAM_BAD_NUMERIC:
    return "expected a proper numerical parameter";
  case PARAM_NEGATIVE_NUMERIC:
    return "expected a positive numerical parameter";
  case PARAM_LIBCURL_DOESNT_SUPPORT:
    return "the installed libcurl version doesn't support this";
  case PARAM_LIBCURL_UNSUPPORTED_PROTOCOL:
    return "a specified protocol is unsupported by libcurl";
  case PARAM_NO_MEM:
    return "out of memory";
  case PARAM_NO_PREFIX:
    return "the given option can't be reversed with a --no- prefix";
  case PARAM_NUMBER_TOO_LARGE:
    return "too large number";
  case PARAM_NO_NOT_BOOLEAN:
    return "used '--no-' for option that isn't a boolean";
  case PARAM_CONTDISP_SHOW_HEADER:
    return "showing headers and --remote-header-name cannot be combined";
  case PARAM_CONTDISP_RESUME_FROM:
    return "--continue-at and --remote-header-name cannot be combined";
  case PARAM_READ_ERROR:
    return "error encountered when reading a file";
  case PARAM_EXPAND_ERROR:
    return "variable expansion failure";
  case PARAM_BLANK_STRING:
    return "blank argument where content is expected";
  default:
    return "unknown error";
  }
}

int SetHTTPrequest(struct OperationConfig *config, HttpReq req, HttpReq *store)
{
  /* this mirrors the HttpReq enum in tool_sdecls.h */
  const char *reqname[]= {
    "", /* unspec */
    "GET (-G, --get)",
    "HEAD (-I, --head)",
    "multipart formpost (-F, --form)",
    "POST (-d, --data)",
    "PUT (-T, --upload-file)"
  };

  if((*store == HTTPREQ_UNSPEC) ||
     (*store == req)) {
    *store = req;
    return 0;
  }
  warnf(config->global, "You can only select one HTTP request method! "
        "You asked for both %s and %s.",
        reqname[req], reqname[*store]);

  return 1;
}

void customrequest_helper(struct OperationConfig *config, HttpReq req,
                          char *method)
{
  /* this mirrors the HttpReq enum in tool_sdecls.h */
  const char *dflt[]= {
    "GET",
    "GET",
    "HEAD",
    "POST",
    "POST",
    "PUT"
  };

  if(!method)
    ;
  else if(curl_strequal(method, dflt[req])) {
    notef(config->global, "Unnecessary use of -X or --request, %s is already "
          "inferred.", dflt[req]);
  }
  else if(curl_strequal(method, "head")) {
    warnf(config->global,
          "Setting custom HTTP method to HEAD with -X/--request may not work "
          "the way you want. Consider using -I/--head instead.");
  }
}
