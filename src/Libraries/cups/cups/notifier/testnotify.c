/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
 * Include necessary headers...
 */

#include <cups/cups-private.h>


/*
 * Local functions...
 */

void	print_attributes(ipp_t *ipp, int indent);


/*
 * 'main()' - Main entry for the test notifier.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line arguments */
     char *argv[])			/* I - Command-line arguments */
{
  int		i;			/* Looping var */
  ipp_t		*event;			/* Event from scheduler */
  ipp_state_t	state;			/* IPP event state */


  setbuf(stderr, NULL);

  fprintf(stderr, "DEBUG: argc=%d\n", argc);
  for (i = 0; i < argc; i ++)
    fprintf(stderr, "DEBUG: argv[%d]=\"%s\"\n", i, argv[i]);
  fprintf(stderr, "DEBUG: TMPDIR=\"%s\"\n", getenv("TMPDIR"));

  for (;;)
  {
    event = ippNew();
    while ((state = ippReadFile(0, event)) != IPP_DATA)
    {
      if (state <= IPP_IDLE)
        break;
    }

    if (state == IPP_ERROR)
      fputs("DEBUG: ippReadFile() returned IPP_ERROR!\n", stderr);

    if (state <= IPP_IDLE)
    {
      ippDelete(event);
      return (0);
    }

    print_attributes(event, 4);
    ippDelete(event);

   /*
    * If the recipient URI is "testnotify://nowait", then we exit after each
    * event...
    */

    if (!strcmp(argv[1], "testnotify://nowait"))
      return (0);
  }
}


/*
 * 'print_attributes()' - Print the attributes in a request...
 */

void
print_attributes(ipp_t *ipp,		/* I - IPP request */
                 int   indent)		/* I - Indentation */
{
  ipp_tag_t		group;		/* Current group */
  ipp_attribute_t	*attr;		/* Current attribute */
  char			buffer[1024];	/* Value buffer */


  for (group = IPP_TAG_ZERO, attr = ipp->attrs; attr; attr = attr->next)
  {
    if ((attr->group_tag == IPP_TAG_ZERO && indent <= 8) || !attr->name)
    {
      group = IPP_TAG_ZERO;
      fputc('\n', stderr);
      continue;
    }

    if (group != attr->group_tag)
    {
      group = attr->group_tag;

      fprintf(stderr, "DEBUG: %*s%s:\n\n", indent - 4, "", ippTagString(group));
    }

    ippAttributeString(attr, buffer, sizeof(buffer));

    fprintf(stderr, "DEBUG: %*s%s (%s%s) %s\n", indent, "", attr->name,
            attr->num_values > 1 ? "1setOf " : "",
	    ippTagString(attr->value_tag), buffer);
  }
}
