/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct node {
	struct node	*next;
	int		value;
};

static void
list_append(struct node **list, int value)
{
	assert(list);

	if (*list == NULL) {
		*list = malloc(sizeof(struct node));
		(*list)->value = value;
		(*list)->next  = NULL;
	} else {
		struct node *n = *list;

		while (n->next != NULL)
			n = n->next;

		n->next = malloc(sizeof(struct node));
		n->next->value = value;
		n->next->next  = NULL;

	}
}

static void
list_manipulate(struct node *list)
{
	while (list) {
		list->value += 1;
		list->value -= 1;
		list = list->next;
	}
}

int
main(void)
{
	struct node *my_list = NULL;

	list_append(&my_list, 0);
	list_append(&my_list, 1);
	list_append(&my_list, 1);
	list_append(&my_list, 2);
	list_append(&my_list, 3);
	list_append(&my_list, 5);
	list_append(&my_list, 8);
	list_append(&my_list, 13);

	for (;;) {
		list_manipulate(my_list);
		sleep(1);
	}

	return 0;
}

