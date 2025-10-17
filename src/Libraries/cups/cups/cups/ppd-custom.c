/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
 * Include necessary headers.
 */

#include "cups-private.h"
#include "ppd-private.h"
#include "debug-internal.h"


/*
 * 'ppdFindCustomOption()' - Find a custom option.
 *
 * @since CUPS 1.2/macOS 10.5@
 */

ppd_coption_t *				/* O - Custom option or NULL */
ppdFindCustomOption(ppd_file_t *ppd,	/* I - PPD file */
                    const char *keyword)/* I - Custom option name */
{
  ppd_coption_t	key;			/* Custom option search key */


  if (!ppd)
    return (NULL);

  strlcpy(key.keyword, keyword, sizeof(key.keyword));
  return ((ppd_coption_t *)cupsArrayFind(ppd->coptions, &key));
}


/*
 * 'ppdFindCustomParam()' - Find a parameter for a custom option.
 *
 * @since CUPS 1.2/macOS 10.5@
 */

ppd_cparam_t *				/* O - Custom parameter or NULL */
ppdFindCustomParam(ppd_coption_t *opt,	/* I - Custom option */
                   const char    *name)	/* I - Parameter name */
{
  ppd_cparam_t	*param;			/* Current custom parameter */


  if (!opt)
    return (NULL);

  for (param = (ppd_cparam_t *)cupsArrayFirst(opt->params);
       param;
       param = (ppd_cparam_t *)cupsArrayNext(opt->params))
    if (!_cups_strcasecmp(param->name, name))
      break;

  return (param);
}


/*
 * 'ppdFirstCustomParam()' - Return the first parameter for a custom option.
 *
 * @since CUPS 1.2/macOS 10.5@
 */

ppd_cparam_t *				/* O - Custom parameter or NULL */
ppdFirstCustomParam(ppd_coption_t *opt)	/* I - Custom option */
{
  if (!opt)
    return (NULL);

  return ((ppd_cparam_t *)cupsArrayFirst(opt->params));
}


/*
 * 'ppdNextCustomParam()' - Return the next parameter for a custom option.
 *
 * @since CUPS 1.2/macOS 10.5@
 */

ppd_cparam_t *				/* O - Custom parameter or NULL */
ppdNextCustomParam(ppd_coption_t *opt)	/* I - Custom option */
{
  if (!opt)
    return (NULL);

  return ((ppd_cparam_t *)cupsArrayNext(opt->params));
}
