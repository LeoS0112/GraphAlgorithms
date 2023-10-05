#include <stdio.h>

void swap(int a[], int x, int y){
    int temp = a[x];
    a[x] = a[y];
    a[y] = temp;
}

void print_int_array(int list[], int length){
    for (int i = 0; i < length; i++){
        printf("%d", list[i]);
    }
}

void insertsort(int list[], int length){
    for (int i=0; i<length; i++){
        int current = i;
        while ((current > 0) && (list[current]<list[current-1])){
            swap(list, current, current-1);
            current = current - 1;
        }
    }
}

void selectsort(int list[], int length){
    for (int i=0; i<length; i++){
        int min = i;
        for (int j=i; j<length; j++){
            if (list[j] < list[min]){
                min = j;
            }
        }
        swap(list, i, min);
    }
}

void binary_insertsort(int list[], int length){
    int min = 0;
    for (int i=0; i<length; i++){
        if (list[i] < list[min]){
            min = i;
        }
    }
    swap(list, 0, min);

    for (int i=0; i<length; i++){
        int upper = i;
        int lower = 0;
        while (upper != lower+1 && upper != lower){
            int middle = (upper + lower) / 2;
            if (list[i] > list[middle]) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        for (int j=0; j<i-upper; j++){
            swap(list, i-j, i-j-1);
        }

    }
}

void merge(int list[], int start, int middle, int end){
    int temporary_space[end - start];
    for (int i=middle; i<end; i++){
        temporary_space[i-middle] = list[i];
    }
    int first_pointer = middle - 1;
    int second_pointer = end - middle - 1;
    for (int i=end-1; i>=0; i--){
        if (first_pointer < 0){
            list[i] = list[second_pointer];
            second_pointer--;
        }
        else if (second_pointer < 0){
            list[i] = list[first_pointer];
            first_pointer--;
        }
        if (list[first_pointer] > temporary_space[second_pointer]){
            list[i] = list[first_pointer];
            first_pointer--;
        } else {
            list[i] = temporary_space[second_pointer];
            second_pointer--;
        }
    }

}
void bottom_up_mergesort(int list[], int length){
    int merge_size = 1;
    while (merge_size <= length){
        int start = 0;
        int middle = start + merge_size;
        int end = middle + merge_size;
        while (middle < length){
            if (end >= length){
                end = length-1;
            }
            merge(list, start, middle, end);
            start = end;
            middle = start + merge_size;
            end = middle + merge_size;
        }
        merge_size = merge_size * 2;
    }
}

int main() {
    int unsorted[10] = {1, 2, 5, 8, 9, 0, 3, 4, 6, 7};
    merge(unsorted, 0, 5, 10);
    print_int_array(unsorted, 10);
    return 0;
}