#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import "bpe.h"

int main (int argc, char *argv[]) {
    if (argc < 2) return puts("Usage: txt2img <text>");
    bpe_context_t tokenizer = bpe_init("vocab.json", "merges.txt");
    int capacity = 77;
    int *ids = calloc(capacity, sizeof(int));

    int size = bpe_encode(tokenizer, argv[1], ids, capacity);
    for (int i = 0; i < capacity; ++i)
        printf("%d ", ids[i]);
    printf("\n");
    bpe_free(tokenizer);

    /*NSLog(@"start");

    NSArray * shape = @[[NSArray alloc]  init];
    MLMultiArrayDataType dataType = MLMultiArrayDataTypeDouble;
    NSError * error = nil;

    MLMultiArray * input =  [MLMultiArray initWithShape:(NSArray*) shape
        dataType:(MLMultiArrayDataType ) dataType
                    error:(NSError **) error];

    MLModel * mymodel = [[MLModel init] initWithContentsOfFile:@"svm.mlmodel"];*/
    return 0;
}
