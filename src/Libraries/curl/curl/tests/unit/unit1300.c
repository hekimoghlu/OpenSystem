/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include "curlcheck.h"

#include "llist.h"

static struct Curl_llist llist;

static struct Curl_llist llist_destination;

static void test_Curl_llist_dtor(void *key, void *value)
{
  /* used by the llist API, does nothing here */
  (void)key;
  (void)value;
}

static CURLcode unit_setup(void)
{
  Curl_llist_init(&llist, test_Curl_llist_dtor);
  Curl_llist_init(&llist_destination, test_Curl_llist_dtor);
  return CURLE_OK;
}

static void unit_stop(void)
{
}

UNITTEST_START
{
  int unusedData_case1 = 1;
  int unusedData_case2 = 2;
  int unusedData_case3 = 3;
  struct Curl_llist_element case1_list;
  struct Curl_llist_element case2_list;
  struct Curl_llist_element case3_list;
  struct Curl_llist_element case4_list;
  struct Curl_llist_element *head;
  struct Curl_llist_element *element_next;
  struct Curl_llist_element *element_prev;
  struct Curl_llist_element *to_remove;
  size_t llist_size = Curl_llist_count(&llist);

  /**
   * testing llist_init
   * case 1:
   * list initiation
   * @assumptions:
   * 1: list size will be 0
   * 2: list head will be NULL
   * 3: list tail will be NULL
   * 4: list dtor will be NULL
  */

  fail_unless(llist.size == 0, "list initial size should be zero");
  fail_unless(llist.head == NULL, "list head should initiate to NULL");
  fail_unless(llist.tail == NULL, "list tail should initiate to NULL");
  fail_unless(llist.dtor == test_Curl_llist_dtor,
               "list dtor should initiate to test_Curl_llist_dtor");

  /**
   * testing Curl_llist_insert_next
   * case 1:
   * list is empty
   * @assumptions:
   * 1: list size will be 1
   * 2: list head will hold the data "unusedData_case1"
   * 3: list tail will be the same as list head
   */

  Curl_llist_insert_next(&llist, llist.head, &unusedData_case1, &case1_list);

  fail_unless(Curl_llist_count(&llist) == 1,
              "List size should be 1 after adding a new element");
  /* test that the list head data holds my unusedData */
  fail_unless(llist.head->ptr == &unusedData_case1,
              "head ptr should be first entry");
  /* same goes for the list tail */
  fail_unless(llist.tail == llist.head,
              "tail and head should be the same");

  /**
   * testing Curl_llist_insert_next
   * case 2:
   * list has 1 element, adding one element after the head
   * @assumptions:
   * 1: the element next to head should be our newly created element
   * 2: the list tail should be our newly created element
   */

  Curl_llist_insert_next(&llist, llist.head,
                         &unusedData_case3, &case3_list);
  fail_unless(llist.head->next->ptr == &unusedData_case3,
              "the node next to head is not getting set correctly");
  fail_unless(llist.tail->ptr == &unusedData_case3,
              "the list tail is not getting set correctly");

  /**
   * testing Curl_llist_insert_next
   * case 3:
   * list has >1 element, adding one element after "NULL"
   * @assumptions:
   * 1: the element next to head should be our newly created element
   * 2: the list tail should different from newly created element
   */

  Curl_llist_insert_next(&llist, llist.head,
                         &unusedData_case2, &case2_list);
  fail_unless(llist.head->next->ptr == &unusedData_case2,
              "the node next to head is not getting set correctly");
  /* better safe than sorry, check that the tail isn't corrupted */
  fail_unless(llist.tail->ptr != &unusedData_case2,
              "the list tail is not getting set correctly");

  /* unit tests for Curl_llist_remove */

  /**
   * case 1:
   * list has >1 element, removing head
   * @assumptions:
   * 1: list size will be decremented by one
   * 2: head will be the head->next
   * 3: "new" head's previous will be NULL
   */

  head = llist.head;
  abort_unless(head, "llist.head is NULL");
  element_next = head->next;
  llist_size = Curl_llist_count(&llist);

  Curl_llist_remove(&llist, llist.head, NULL);

  fail_unless(Curl_llist_count(&llist) ==  (llist_size-1),
               "llist size not decremented as expected");
  fail_unless(llist.head == element_next,
               "llist new head not modified properly");
  abort_unless(llist.head, "llist.head is NULL");
  fail_unless(llist.head->prev == NULL,
              "new head previous not set to null");

  /**
   * case 2:
   * removing non head element, with list having >=2 elements
   * @setup:
   * 1: insert another element to the list to make element >=2
   * @assumptions:
   * 1: list size will be decremented by one ; tested
   * 2: element->previous->next will be element->next
   * 3: element->next->previous will be element->previous
   */
  Curl_llist_insert_next(&llist, llist.head, &unusedData_case3,
                         &case4_list);
  llist_size = Curl_llist_count(&llist);
  fail_unless(llist_size == 3, "should be 3 list members");

  to_remove = llist.head->next;
  abort_unless(to_remove, "to_remove is NULL");
  element_next = to_remove->next;
  element_prev = to_remove->prev;
  Curl_llist_remove(&llist, to_remove, NULL);
  fail_unless(element_prev->next == element_next,
              "element previous->next is not being adjusted");
  abort_unless(element_next, "element_next is NULL");
  fail_unless(element_next->prev == element_prev,
              "element next->previous is not being adjusted");

  /**
   * case 3:
   * removing the tail with list having >=1 element
   * @assumptions
   * 1: list size will be decremented by one ;tested
   * 2: element->previous->next will be element->next ;tested
   * 3: element->next->previous will be element->previous ;tested
   * 4: list->tail will be tail->previous
   */

  to_remove = llist.tail;
  element_prev = to_remove->prev;
  Curl_llist_remove(&llist, to_remove, NULL);
  fail_unless(llist.tail == element_prev,
              "llist tail is not being adjusted when removing tail");

  /**
   * case 4:
   * removing head with list having 1 element
   * @assumptions:
   * 1: list size will be decremented by one ;tested
   * 2: list head will be null
   * 3: list tail will be null
   */

  to_remove = llist.head;
  Curl_llist_remove(&llist, to_remove, NULL);
  fail_unless(llist.head == NULL,
              "llist head is not NULL while the llist is empty");
  fail_unless(llist.tail == NULL,
              "llist tail is not NULL while the llist is empty");

  Curl_llist_destroy(&llist, NULL);
  Curl_llist_destroy(&llist_destination, NULL);
}
UNITTEST_STOP
