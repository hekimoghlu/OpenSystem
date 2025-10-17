/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#ifdef TEST
# include <stdio.h>
#endif
#include <string.h>
#include <stdlib.h>
#include "strlist.h"


/*****************************************************************
**	prepstrlist (str, delim)
**	prepare a string with delimiters to a so called strlist.
**	'str' is a list of substrings delimited by 'delim'
**	The # of strings is stored at the first byte of the allocated
**	memory. Every substring is stored as a '\0' terminated C-String.
**	The function returns a pointer to dynamic allocated memory
*****************************************************************/
char	*prepstrlist (const char *str, const char *delim)
{
	char	*p;
	char	*new;
	int	len;
	int	cnt;

	if ( str == NULL )
		return NULL;

	len = strlen (str);
	if ( (new = malloc (len + 2)) == NULL )
		return new;

	cnt = 0;
	p = new;
	for ( *p++ = '\0'; *str; str++ )
	{
		if ( strchr (delim, *str) == NULL )
			*p++ = *str;
		else if ( p[-1] != '\0' )
		{
			*p++ = '\0';
			cnt++;
		}
	}
	*p = '\0';	/*terminate string */
	if ( p[-1] != '\0' )
		cnt++;
	*new = cnt & 0xFF;

	return new;
}

/*****************************************************************
**	isinlist (str, list)
**	check if 'list' contains 'str'
*****************************************************************/
int	isinlist (const char *str, const char *list)
{
	int	cnt;

	if ( list == NULL || *list == '\0' )
		return 1;
	if ( str == NULL || *str == '\0' )
		return 0;

	cnt = *list;
	while ( cnt-- > 0 )
	{
		list++;
		if ( strcmp (str, list) == 0 )
			return 1;
		list += strlen (list);
	}

	return 0;
}

/*****************************************************************
**	unprepstrlist (list, delimc)
*****************************************************************/
char	*unprepstrlist (char *list, char delimc)
{
	char	*p;
	int	cnt;

	cnt = *list & 0xFF;
	p = list;
	for ( *p++ = delimc; cnt > 1; p++ )
		if ( *p == '\0' )
		{
			*p = delimc;
			cnt--;
		}

	return list;
}

#ifdef TEST
main (int argc, char *argv[])
{
	FILE	*fp;
	char	*p;
	char	*searchlist = NULL;
	char	group[255];

	if ( argc > 1 )
		searchlist = prepstrlist (argv[1], LISTDELIM);

	printf ("searchlist: %d entrys: \n", searchlist[0]);
	if ( (fp = fopen ("/etc/group", "r")) == NULL )
		exit (fprintf (stderr, "can't open file\n"));

	while ( fscanf (fp, "%[^:]:%*[^\n]\n", group) != EOF )
		if ( isinlist (group, searchlist) )
			printf ("%s\n", group);

	fclose (fp);

	printf ("searchlist: \"%s\"\n", unprepstrlist  (searchlist, *LISTDELIM));
	for ( p = searchlist; *p; p++ )
		if ( *p < 32 )
			printf ("<%d>", *p);
		else
			printf ("%c", *p);
	printf ("\n");
}
#endif
