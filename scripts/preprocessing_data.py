import os
import sys
import nltk
import json
import enchant
from nltk.corpus import wordnet as wn

current_dir = os.getcwd()
os.chdir('../')


## Loading data
image_data = json.load(open('image_data.json'))
print('image data length: ' + str(len(image_data)))
relationships_data = json.load(open('relationships.json'))
print('relationship data length: ' + str(len(relationships_data)))

## The subject and object should be none
en_dict = enchant.Dict("en_US")
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}



relationship_count = 0
predicate_dataset = {}

spelling_error_counter = 0
length_matching_counter = 0
# word_mismatch_counter = 0

relationships = {}

for d_id,rs in enumerate(relationships_data):
    im_relationships = {}
    for r_id,r in enumerate(rs['relationships']):
        try:
            normalized_predicate = '_'.join([nltk.stem.WordNetLemmatizer().lemmatize(x, 'v') for x in
                                             r['predicate'].strip('.').strip(',').encode('ascii', 'replace').split()])
            normalized_subject = '_'.join([nltk.stem.WordNetLemmatizer().lemmatize(x, 'n') for x in
                                           r['subject']['name'].strip('.').strip(',').encode('ascii', 'replace').split()])
            normalized_object = '_'.join([nltk.stem.WordNetLemmatizer().lemmatize(x, 'n') for x in
                                           r['object']['name'].strip('.').strip(',').encode('ascii', 'replace').split()])

            if (not en_dict.check(normalized_predicate.replace('_', '-'))) or \
                    (not en_dict.check(normalized_subject.replace('_', '-'))) or \
                    (not en_dict.check(normalized_object.replace('_', '-'))):
                spelling_error_counter += 1
                # print('Wrong spelling({}):{}-{}-{}\n'.format(spelling_error_counter, normalized_subject, normalized_predicate, normalized_object));
                continue

            normalized_predicate = normalized_predicate.lower().replace('-', '_')
            normalized_subject = normalized_subject.lower().replace('-', '_')
            normalized_object = normalized_object.lower().replace('-', '_')

            if len(normalized_predicate) <= 1 or len(normalized_subject) <=1 or len(normalized_object) <=1:
                length_matching_counter += 1
                # print('length not matched:{}-{}-{}\n'.format(r['subject']['name'], r['predicate'], r['object']['name']))
                continue

            # if normalized_object not in nouns or normalized_subject not in nouns:
            #     # print('Subject or Object no in Nouns:{}-{}-{}\n'.format(r['subject']['name'], r['predicate'], r['object']['name']))
            #     word_mismatch_counter += 1
            #     continue
            relationship_item = {}
            relationship_item['object'] = normalized_object
            relationship_item['subject'] = normalized_subject
            relationship_item['sub_box'] = \
                    (r['subject']['x'], r['subject']['y'], r['subject']['x'] + r['subject']['w'], \
                     r['subject']['y'] + r['subject']['h'])
            relationship_item['obj_box'] = \
                    (r['object']['x'], r['object']['y'], r['object']['x'] + r['object']['w'], \
                     r['object']['y'] + r['object']['h'])
            relationship_item['predicate'] = normalized_predicate
            if 'relationships' not in im_relationships.keys():
                im_relationships['relationships'] = [relationship_item]
            else:
                im_relationships['relationships'].append(relationship_item)
            relationship_count += 1
        except Exception as inst:
            print inst
            print d_id
            print r_id
            # raw_input("Press Enter to continue...")
            print('({}, {}): [{}]-[{}]-[{}]\n'.format(d_id, r_id, r['subject']['name'], r['predicate'], r['object']['name']))
            print('Error: [{}]-[{}]-[{}]\n'.format(r['subject']['name'].encode('ascii', 'replace'), r['predicate'].encode('ascii', 'replace'), r['object']['name'].encode('ascii', 'replace')))
            #  raw_input('Press Enter to continue...')
            pass
    if d_id%5000 == 0:
        print(str(d_id) + ' images processed, ' + str(relationship_count) + ' relationships')

    if 'relationships' in im_relationships.keys():
        im_relationships['path'] = str(image_data[d_id]['image_id']) + '.jpg'
        im_relationships['width'] = image_data[d_id]['width']
        im_relationships['height'] = image_data[d_id]['height']
        relationships[d_id] = im_relationships

del relationships_data
print('Currently, we have ' + str(relationship_count) + ' relationship tuples and {} images'.format(len(relationships.keys())))
print('Spelling error: {}'.format(spelling_error_counter))
print('Length matching error: {}'.format(length_matching_counter))
# print('word mismatch error: {}'.format(word_mismatch_counter))

if __name__ == "__main__":
    def output_annoatation(output_path, relationships):
        with open(output_path, 'w') as f:
            output_counter = 0
            for item_key in relationships:
                im_item = relationships[item_key]
                f.write('# {}\n'.format(item_key))  # to output the item key
                f.write(im_item['path'] + '\n')  # to output the image path
                f.write('{}\n{}\n'.format(im_item['height'], im_item['width']))  # output the height and width
                f.write('{}\n'.format(len(im_item['relationships'])))
                for r in im_item['relationships']:  # output the relationship item [subject]-[predicate]-[object]-[sub_box]-[obj_box]
                    f.write(r['subject'].replace(' ', '_'))
                    f.write(' ' + r['predicate'].replace(' ', '_'))
                    f.write(' ' + r['object'].replace(' ', '_'))
                    for item in r['sub_box']:
                        f.write(' ' + str(item))
                    for item in r['obj_box']:
                        f.write(' ' + str(item))
                    f.write('\n')
                output_counter += 1
                if output_counter % 1000 == 0:
                    print('{}/{} images processed'.format(output_counter, len(relationships.keys())))

        print('Result output to: {}'.format(output_path))


    os.chdir(current_dir)
    output_annoatation('output/filtered_relationship.txt', relationships)



os.chdir(current_dir)
