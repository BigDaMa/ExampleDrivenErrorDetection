import os

out_path = "/home/felix/data_more/new/out/"

#with open('/home/felix/data_more/new/test.txt') as fp:
with open('/home/felix/data_more/new/infobox_properties_mapped_en.ttl') as fp:
    for line in fp:
        tokens = line.split('<')

        if len(tokens) > 1:
            print line
            resource = tokens[1].split("/")[-1].split(">")[0].replace("_", " ").upper()

            if '(' in resource:
                resource = resource.split("(")[0].strip().upper()


            print resource


            if '"^^' in tokens[2]:
                new_tokens = tokens[2].split('"')
                property = new_tokens[0].split("/")[-1].split(">")[0].upper()
                value = new_tokens[1].upper()

            else:
                property = tokens[2].split("/")[-1].split(">")[0].upper()
                value = tokens[3].split("/")[-1].split(">")[0].replace("_", " ").upper()

                if '(' in value:
                    value = value.split("(")[0].strip().upper()

            print property
            print value

            filename = out_path + property + ".rel.txt"


            try:
                with open(filename, "a") as myfile:
                    myfile.write(resource + "\t" + property + "\t" + value +"\n")
                myfile.close()
            except IOError:
                print "warning"


        #print tokens